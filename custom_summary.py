import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize the OpenAI client
#openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key in environment variables

client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

def get_response(client, query):
    """Query the LLM to generate a response."""
    response = client.chat.completions.create(
    model='Meta-Llama-3.1-8B-Instruct',
    messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":f"{query}"}],
    temperature =  0.1,
    top_p = 0.1
    )
    return response.choices[0].message.content


def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Splits text into manageable chunks using a recursive splitter."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def summarize_chunks(client, chunks):
    """Generates a summary for each chunk."""
    summaries = []
    for idx, chunk in enumerate(chunks):
        query = f"Please summarize the following text:\n\n{chunk}"
        summary = get_response(client, query)
        summaries.append(summary)
        print(f"Chunk {idx + 1}/{len(chunks)} summarized.")
    return summaries


def recursive_summarization(client, text, chunk_size=2000, chunk_overlap=200):
    """Recursively summarize text until it fits within the context length."""
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    
    print(f"Generated {len(chunks)} chunks. Starting summarization...")
    summaries = summarize_chunks(client, chunks)
    
    combined_summary = " ".join(summaries)
    
    if len(split_text_into_chunks(combined_summary, chunk_size, chunk_overlap)) > 1:
        print("Combined summary exceeds context length. Recursing...")
        return recursive_summarization(client, combined_summary, chunk_size, chunk_overlap)
    
    print("Final summary generated.")
    return combined_summary


def fetch_document(url):
    """Fetches a document from the specified URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Combine all document texts if there are multiple
    return " ".join(doc.page_content for doc in docs)


# Example usage
if __name__ == "__main__":
    # url = "https://en.wikipedia.org/wiki/Sea_surface_temperature"
    # print("Fetching document...")
    # document = fetch_document(url)
    
    document="""
    The Evolution of Technology: A Philosophical Exploration

Technology, in its myriad forms, has been an intrinsic part of human civilization, guiding the course of history, shaping societies, and redefining the limits of what we perceive as possible. From the invention of the wheel to the rise of artificial intelligence, every breakthrough has been a testament to humanity's quest for efficiency, understanding, and mastery over the environment.

The Birth of Innovation
The early humans' use of tools marked the dawn of technology. Tools made of stone, bone, and wood represented the ingenuity of early societies. The wheel, for instance, fundamentally altered the dynamics of transport and commerce. It is said that early Mesopotamians invented the wheel around 3500 BCE, initially used for pottery before revolutionizing transportation. The simple concept of reducing friction by allowing objects to roll instead of slide laid the groundwork for modern vehicles.

Fire, another cornerstone of technological development, not only served as a source of light and heat but also became a symbol of community, protection, and culinary experimentation. The controlled use of fire transformed dietary habits, enabling the consumption of cooked food, which is believed to have had profound effects on human physiology and cognition.

The Agricultural Revolution
Agriculture marked a major turning point in the history of technology. The domestication of plants and animals allowed societies to settle, leading to the establishment of villages and cities. Tools such as the plow amplified productivity, while innovations in irrigation systems enabled the expansion of arable land. The ancient Egyptians, for instance, developed sophisticated methods of channeling the Nile's waters to sustain agriculture, giving rise to one of history's great civilizations.

The Industrial Era
Fast forward to the Industrial Revolution, where steam engines, mechanized looms, and factories heralded a new era. The shift from agrarian economies to industrialized societies brought unparalleled growth and challenges. Coal became the lifeblood of the industrial world, fueling factories, trains, and ships. This period also saw the rise of mass production, facilitated by assembly lines—a concept that would later be perfected by Henry Ford.

While the Industrial Revolution brought material wealth and advancements in medicine and infrastructure, it also raised ethical and environmental questions. The exploitation of workers, particularly in coal mines and textile mills, led to movements advocating labor rights. Meanwhile, deforestation and pollution signaled the cost of progress.

The Digital Transformation
The 20th century witnessed the birth of computing. Alan Turing, often regarded as the father of modern computing, envisioned machines capable of processing information, solving problems, and mimicking human intelligence. His theories laid the foundation for the development of the modern computer.

By the late 20th century, personal computers became a household staple. The introduction of the internet further revolutionized communication, commerce, and culture. E-commerce giants like Amazon and platforms like Google and Facebook reshaped global economies, while social media changed how individuals interact and express themselves.

Artificial Intelligence: The Frontier
Today, artificial intelligence (AI) represents the cutting edge of technology. Machine learning algorithms power applications from autonomous vehicles to personalized recommendations on streaming platforms. AI holds the promise of revolutionizing healthcare, with predictive models enabling earlier diagnoses and treatment plans tailored to individual patients.

However, AI also poses ethical dilemmas. The automation of jobs raises concerns about unemployment and economic inequality. Moreover, questions about data privacy and the ethical use of algorithms persist. How do we ensure that AI systems are unbiased and transparent? How do we prevent misuse in areas like surveillance and warfare?

Technology and Society
As technology advances, its societal impacts become increasingly complex. Consider the smartphone, a device that has become almost an extension of the human body. While it has revolutionized access to information and connectivity, it has also been linked to issues like digital addiction and mental health challenges.

Similarly, social media platforms, while democratizing content creation, have also contributed to the spread of misinformation and echo chambers. The question arises: how do we balance the benefits of technology with its potential harms?

Philosophical Reflections
Philosophers throughout history have pondered the role of technology in human life. Martin Heidegger, for instance, warned against viewing technology purely as a means to an end. He argued that such a perspective could obscure the essence of being and the natural world. Instead, he urged a more reflective engagement with technology.

Marshall McLuhan famously stated, "The medium is the message," highlighting how technological mediums themselves, rather than their content, shape human experience. For example, the rise of television fundamentally altered societal dynamics by prioritizing visual communication over text.

The Road Ahead
Looking to the future, emerging technologies like quantum computing, gene editing, and renewable energy promise to tackle some of humanity's greatest challenges. Quantum computers could solve problems currently deemed insurmountable, while advancements in CRISPR technology could eradicate genetic disorders.

Yet, the path forward is fraught with uncertainties. As humanity wields more powerful tools, the stakes grow higher. Climate change, fueled by industrialization, requires urgent technological interventions, from carbon capture to sustainable energy sources.

Conclusion
Technology is neither inherently good nor evil—it is a tool shaped by the intentions of its creators and users. As we navigate an era of rapid change, it is imperative to approach technology with both enthusiasm and caution. By fostering ethical innovation and equitable access, humanity can harness the power of technology to build a better future.
The Evolution of Technology: A Philosophical Exploration

Technology, in its myriad forms, has been an intrinsic part of human civilization, guiding the course of history, shaping societies, and redefining the limits of what we perceive as possible. From the invention of the wheel to the rise of artificial intelligence, every breakthrough has been a testament to humanity's quest for efficiency, understanding, and mastery over the environment.

The Birth of Innovation
The early humans' use of tools marked the dawn of technology. Tools made of stone, bone, and wood represented the ingenuity of early societies. The wheel, for instance, fundamentally altered the dynamics of transport and commerce. It is said that early Mesopotamians invented the wheel around 3500 BCE, initially used for pottery before revolutionizing transportation. The simple concept of reducing friction by allowing objects to roll instead of slide laid the groundwork for modern vehicles.

Fire, another cornerstone of technological development, not only served as a source of light and heat but also became a symbol of community, protection, and culinary experimentation. The controlled use of fire transformed dietary habits, enabling the consumption of cooked food, which is believed to have had profound effects on human physiology and cognition.

The Agricultural Revolution
Agriculture marked a major turning point in the history of technology. The domestication of plants and animals allowed societies to settle, leading to the establishment of villages and cities. Tools such as the plow amplified productivity, while innovations in irrigation systems enabled the expansion of arable land. The ancient Egyptians, for instance, developed sophisticated methods of channeling the Nile's waters to sustain agriculture, giving rise to one of history's great civilizations.

The Industrial Era
Fast forward to the Industrial Revolution, where steam engines, mechanized looms, and factories heralded a new era. The shift from agrarian economies to industrialized societies brought unparalleled growth and challenges. Coal became the lifeblood of the industrial world, fueling factories, trains, and ships. This period also saw the rise of mass production, facilitated by assembly lines—a concept that would later be perfected by Henry Ford.

While the Industrial Revolution brought material wealth and advancements in medicine and infrastructure, it also raised ethical and environmental questions. The exploitation of workers, particularly in coal mines and textile mills, led to movements advocating labor rights. Meanwhile, deforestation and pollution signaled the cost of progress.

The Digital Transformation
The 20th century witnessed the birth of computing. Alan Turing, often regarded as the father of modern computing, envisioned machines capable of processing information, solving problems, and mimicking human intelligence. His theories laid the foundation for the development of the modern computer.

By the late 20th century, personal computers became a household staple. The introduction of the internet further revolutionized communication, commerce, and culture. E-commerce giants like Amazon and platforms like Google and Facebook reshaped global economies, while social media changed how individuals interact and express themselves.

Artificial Intelligence: The Frontier
Today, artificial intelligence (AI) represents the cutting edge of technology. Machine learning algorithms power applications from autonomous vehicles to personalized recommendations on streaming platforms. AI holds the promise of revolutionizing healthcare, with predictive models enabling earlier diagnoses and treatment plans tailored to individual patients.

However, AI also poses ethical dilemmas. The automation of jobs raises concerns about unemployment and economic inequality. Moreover, questions about data privacy and the ethical use of algorithms persist. How do we ensure that AI systems are unbiased and transparent? How do we prevent misuse in areas like surveillance and warfare?

Technology and Society
As technology advances, its societal impacts become increasingly complex. Consider the smartphone, a device that has become almost an extension of the human body. While it has revolutionized access to information and connectivity, it has also been linked to issues like digital addiction and mental health challenges.

Similarly, social media platforms, while democratizing content creation, have also contributed to the spread of misinformation and echo chambers. The question arises: how do we balance the benefits of technology with its potential harms?

Philosophical Reflections
Philosophers throughout history have pondered the role of technology in human life. Martin Heidegger, for instance, warned against viewing technology purely as a means to an end. He argued that such a perspective could obscure the essence of being and the natural world. Instead, he urged a more reflective engagement with technology.

Marshall McLuhan famously stated, "The medium is the message," highlighting how technological mediums themselves, rather than their content, shape human experience. For example, the rise of television fundamentally altered societal dynamics by prioritizing visual communication over text.

The Road Ahead
Looking to the future, emerging technologies like quantum computing, gene editing, and renewable energy promise to tackle some of humanity's greatest challenges. Quantum computers could solve problems currently deemed insurmountable, while advancements in CRISPR technology could eradicate genetic disorders.

Yet, the path forward is fraught with uncertainties. As humanity wields more powerful tools, the stakes grow higher. Climate change, fueled by industrialization, requires urgent technological interventions, from carbon capture to sustainable energy sources.

Conclusion
Technology is neither inherently good nor evil—it is a tool shaped by the intentions of its creators and users. As we navigate an era of rapid change, it is imperative to approach technology with both enthusiasm and caution. By fostering ethical innovation and equitable access, humanity can harness the power of technology to build a better future.



    """
    
    print("Starting recursive summarization...")
    final_summary = recursive_summarization(client, document)
    
    print("\nFinal Summary:")
    print(final_summary)
