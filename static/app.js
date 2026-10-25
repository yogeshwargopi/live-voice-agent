const els = {
  pills: document.querySelectorAll(".pill"),
  log: document.getElementById("log"),
  empty: document.getElementById("empty"),
  transcript: document.getElementById("transcript"),
  status: document.getElementById("status"),
  error: document.getElementById("error"),
  mic: document.getElementById("mic"),
  label: document.getElementById("mic-label"),
  clear: document.getElementById("clear"),
};

const SESSION_KEY = "friend-session";

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `s-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = newId();
  localStorage.setItem(SESSION_KEY, sessionId);
}

let language = "auto";
let recorder = null;
let chunks = [];
let stream = null;
let player = null;
let endPlayback = null;
let turn = 0;
let recording = false;
let busy = false;
let blocked = false;
let discardRecording = false;

function setStatus(text) {
  els.status.textContent = text;
}

function setError(text) {
  if (!text) {
    els.error.hidden = true;
    els.error.textContent = "";
    return;
  }
  els.error.hidden = false;
  els.error.textContent = text;
}

function renderMic() {
  const thinking = busy && !recording;
  els.mic.disabled = blocked || thinking;
  els.mic.classList.toggle("is-listening", recording);
  els.mic.setAttribute("aria-pressed", recording ? "true" : "false");
  els.mic.setAttribute("aria-label", recording ? "Stop and send" : "Talk");
  els.label.textContent = recording ? "Stop" : "Talk";
}

function hideEmpty() {
  els.empty.hidden = true;
}

function showEmpty() {
  els.transcript.replaceChildren();
  els.empty.hidden = false;
}

function addBubble(role, text, langName) {
  hideEmpty();
  const item = document.createElement("article");
  item.className = `bubble ${role}`;
  const who = document.createElement("p");
  who.className = "who";
  const name = langName ? ` · ${langName}` : "";
  who.textContent = role === "user" ? `You${name}` : `Friend${name}`;
  const said = document.createElement("p");
  said.className = "said";
  said.textContent = text;
  item.append(who, said);
  els.transcript.append(item);
  els.log.scrollTop = els.log.scrollHeight;
}

function stopPlayback() {
  const done = endPlayback;
  endPlayback = null;
  if (player) {
    player.onended = null;
    player.onerror = null;
    player.pause();
    if (player._url) URL.revokeObjectURL(player._url);
    player = null;
  }
  if (done) done();
}

function releaseMic() {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  stream = null;
}

function mimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const type of types) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function playAudio(b64, myTurn) {
  return new Promise((resolve) => {
    if (myTurn !== turn) {
      resolve();
      return;
    }
    endPlayback = resolve;
    const bytes = Uint8Array.from(atob(b64), (char) => char.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/mpeg" }));
    player = new Audio(url);
    player._url = url;
    setStatus("Speaking");
    const finish = (failed) => {
      if (endPlayback === resolve) endPlayback = null;
      if (player && player._url) URL.revokeObjectURL(player._url);
      player = null;
      if (failed && myTurn === turn) {
        setError("The reply is on screen, but the audio didn't play.");
      }
      if (myTurn === turn && !recording && !busy) setStatus("");
      resolve();
    };
    player.onended = () => finish(false);
    player.onerror = () => finish(true);
    player.play().catch(() => finish(true));
  });
}

async function sendTurn(blob) {
  const myTurn = turn;
  busy = true;
  setStatus("Thinking");
  setError("");
  renderMic();
  const body = new FormData();
  const type = blob.type || "audio/webm";
  const ext = type.includes("mp4") || type.includes("m4a") || type.includes("aac") ? "m4a" : "webm";
  body.append("audio", blob, `speech.${ext}`);
  body.append("session_id", sessionId);
  body.append("language", language);
  try {
    const res = await fetch("/api/talk", { method: "POST", body });
    const data = await res.json();
    if (myTurn !== turn) return;
    if (data.user_text) addBubble("user", data.user_text, data.user_language_name || "");
    if (data.assistant_text) addBubble("assistant", data.assistant_text, data.language_name || "");
    if (!res.ok && !data.assistant_text) {
      setError(data.error || "Something went wrong. Try again.");
      setStatus("");
      return;
    }
    setError(data.error || "");
    busy = false;
    renderMic();
    if (data.audio_base64 && myTurn === turn) {
      await playAudio(data.audio_base64, myTurn);
    } else if (myTurn === turn) {
      setStatus("");
    }
  } catch {
    if (myTurn !== turn) return;
    setError("Couldn't reach the assistant. Check that it is still running.");
    setStatus("");
  } finally {
    if (myTurn === turn) {
      busy = false;
      renderMic();
    }
  }
}

async function startRecording() {
  if (!window.MediaRecorder || !navigator.mediaDevices) {
    setError("This browser can't record audio. Open the page in Chrome or Safari.");
    return;
  }
  setError("");
  stopPlayback();
  turn += 1;
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true },
  });
  const type = mimeType();
  chunks = [];
  recorder = type ? new MediaRecorder(stream, { mimeType: type }) : new MediaRecorder(stream);
  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size) chunks.push(event.data);
  };
  recorder.onstop = () => {
    const blobType = recorder.mimeType || type || "audio/webm";
    const blob = new Blob(chunks, { type: blobType });
    recording = false;
    releaseMic();
    renderMic();
    if (discardRecording) {
      discardRecording = false;
      busy = false;
      setStatus("");
      renderMic();
      return;
    }
    sendTurn(blob);
  };
  recorder.start(250);
  recording = true;
  setStatus("Listening");
  renderMic();
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") {
    busy = true;
    setStatus("Thinking");
    renderMic();
    recorder.stop();
  }
}

els.mic.addEventListener("click", async () => {
  if (blocked) return;
  try {
    if (recording) {
      stopRecording();
      return;
    }
    if (busy) return;
    await startRecording();
  } catch {
    recording = false;
    releaseMic();
    setStatus("");
    setError("Allow the microphone for this page, then press Talk again.");
    renderMic();
  }
});

els.clear.addEventListener("click", async () => {
  turn += 1;
  stopPlayback();
  if (recording && recorder && recorder.state !== "inactive") {
    discardRecording = true;
    recorder.stop();
  }
  recording = false;
  busy = false;
  showEmpty();
  setStatus("");
  if (!blocked) setError("");
  renderMic();
  try {
    await fetch("/api/clear", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ session_id: sessionId }),
    });
  } catch {
    if (!blocked) setError("Couldn't clear the chat. Check that the assistant is still running.");
  }
});

for (const pill of els.pills) {
  pill.addEventListener("click", () => {
    language = pill.dataset.lang;
    for (const other of els.pills) {
      const on = other === pill;
      other.classList.toggle("is-on", on);
      other.setAttribute("aria-checked", on ? "true" : "false");
    }
  });
}

async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (!data.ok) {
      blocked = true;
      setError(data.error || "Add your OpenAI API key to the .env file, then refresh this page.");
    } else {
      blocked = false;
    }
  } catch {
    blocked = true;
    setError("The assistant isn't running. Start it with ./start.sh.");
  }
  renderMic();
}

blocked = true;
renderMic();
checkStatus();
