import os
import openai


def get_response(client,query):
    response = client.chat.completions.create(
    model='Meta-Llama-3.1-8B-Instruct',
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":f"{query}"}],
    temperature =  0.1,
    top_p = 0.1
    )

    return response.choices[0].message.content
      



