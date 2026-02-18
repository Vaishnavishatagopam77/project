# Deployment Guide

You can now run the **entire application** (frontend and backend) with a single command.

## How to Run locally (Production Mode)

1. Open a terminal in the project root:
   `c:\Users\SYASH\Downloads\NeoVihar-AI-learnging-agent-main\NeoVihar-AI-learnging-agent-main`

2. Run the application:
   ```bash
   python main.py
   ```

3. Open your browser to:
   **http://localhost:8000**

   (You no longer need to run `npm run dev` or use port 5173).

## How to Deploy to the Cloud (e.g. Render/Railway)

Since the frontend is now served by the backend, you can deploy this as a single **Python Web Service**.

1. **Requirements**: Ensure `requirements.txt` includes:
   `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `python-jose`, `passlib`, `bcrypt`, `python-dotenv`, `openai`

2. **Build Command**: 
   Most platforms will just run `pip install -r requirements.txt`. 
   *Note: You must ensure the `frontend/dist` folder is included in your upload/git repo, OR adds a build step to run `npm install && npm run build` before starting python.*

3. **Start Command**:
   `python main.py` or `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Updating the Frontend
If you make changes to the React code in `/frontend`, you must rebuild it for them to appear in the python app:

```bash
cd frontend
npm run build
cd ..
python main.py
```
