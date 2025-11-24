from openai import OpenAI
import base64


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:1234/v1" 
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def vqa_inference(image_path: str, question: str) -> str:
    image_data = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="5CD-AI/Vintern-1B-v3_5",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }],
        temperature=0.01,
        top_p=0.1,
        max_tokens=512,
        extra_body={
            "min_p": 0.1,
            "top_k": 1,
            "repetition_penalty": 1.1,
            "best_of": 1,
        }
    )
    
    return response.choices[0].message.content
