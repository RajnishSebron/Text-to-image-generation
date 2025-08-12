from fastapi import FastAPI, Response
from src.inference import ImageGenerator
import io

app = FastAPI()
generator = ImageGenerator("./models/finetuned")

@app.post("/generate")
async def generate_image(prompt: str):
    image = generator.generate_image(prompt)
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")
