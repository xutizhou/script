import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

prompt = "The quick brown fox jumps over the lazy dog."

# response = client.chat.completions.create(
#     model="default",
#     prompt=prompt,
#     max_tokens=10,
# )

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response)

