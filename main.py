import uvicorn as uv

if __name__ == '__main__':
    uv.run("src.app:app", host="0.0.0.0", port=5000, reload=True)