# Friend

A local voice companion for English, Hindi, Tamil, and Telugu. It only talks. Put your OpenAI API key in `.env` and the rest is ready.

## Run

1. Copy `.env.example` to `.env` and paste your OpenAI API key.
2. Run `./start.sh`.
3. Open http://127.0.0.1:8000 and allow the microphone.

Press Talk, speak, then press Stop. The reply is spoken in the language you used. Leave the language on Auto, or lock English, Hindi, Tamil, or Telugu.

Hindi and Tamil speech uses OpenAI's published voices. Telugu is understood and answered in Telugu; the spoken voice can sound less natural.
