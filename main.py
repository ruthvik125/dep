import streamlit as st
import uuid
import os
import base64
import openai
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
import re
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

st.set_page_config(page_title='Chat + Mermaid UI', page_icon='🤖', layout="wide")


# Load environment variables
#load_dotenv()


# Initialize OpenAI client
client = openai.OpenAI(
    api_key=st.secrets["SAMBANOVA_API_KEY"]
    ,#os.environ.get("SAMBANOVA_API_KEY"),
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














# Function to extract Mermaid code from the response
def extract_mermaid_code(response):
    match = re.search(r"```mermaid\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Function to generate Mermaid graph
def generate_mermaid_graph(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    image_url = "https://mermaid.ink/img/" + base64_string
    
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        st.error(f"Failed to fetch image. HTTP Status Code: {response.status_code}")
        return None

# Function to fetch document content from a URL
def fetch_document(url):
    """Fetches a document from the specified URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return " ".join(doc.page_content for doc in docs)

# Function to handle user input and display chat history
def handle_userinput(user_question: str,diagram_type_option:str, input_data: str = None) -> None:
    if user_question:
        try:
            with st.spinner('🤖 Summarizing the input...'):
                # Summarize input data (code or document content)
                summary = recursive_summarization(client,input_data) if input_data else None
                #st.info(f"Summary:\n{summary}")

            
            modified_question_for_diagram_type = (
                f"For this summary, which Mermaid diagram is suitable?\n{summary}\n"
                "Options: Flowchart, Sequence Diagram, Class Diagram, State Diagram, Entity Relationship Diagram, "
                "User Journey, Gantt, Pie Chart, Quadrant Chart, Requirement Diagram, Gitgraph (Git) Diagram, "
                "C4 Diagram, Mindmaps, Timeline, ZenUML, Sankey, XY Chart, Block Diagram, Packet, Kanban, Architecture"
            )
           

# Modify the logic to determine the diagram type
            if diagram_type_option == "Automatic Detection":
                with st.spinner('🤖 Determining the diagram type...'):
                    modified_question_for_diagram_type = (
                        f"For this summary, which Mermaid diagram is suitable?\n{summary}\n"
                        "Options: Flowchart, Sequence Diagram, Class Diagram, State Diagram, Entity Relationship Diagram, "
                        "User Journey, Gantt, Pie Chart, Quadrant Chart, Requirement Diagram, Gitgraph (Git) Diagram, "
                        "C4 Diagram, Mindmaps, Timeline, ZenUML, Sankey, XY Chart, Block Diagram, Packet, Kanban, Architecture"
                    )
                    resp_text = get_response(client, modified_question_for_diagram_type)
                    diagram_type = resp_text.strip()  # Infer the type from the model
            else:
                diagram_type = diagram_type_option  # Use the type selected by the user

            
            
            # Modify user question to include the summary for generating a Mermaid graph
            modified_question = (
                f"{user_question}\nBased on the following summary, generate a Mermaid {diagram_type}:\n{summary}.OUTPUT must only contain the mermaid code with proper formatting for the syntax"
            )

            with st.spinner('🤖 Generating response...'):
                # Get response from model
                resp_text = get_response(client, modified_question)
                response = {"answer": f"{resp_text}"}

            # Extract and generate Mermaid graph
            mermaid_code = extract_mermaid_code(response['answer'])
            if mermaid_code:
                st.info("Rendering graph...")
                graph_image = generate_mermaid_graph(mermaid_code)
                if graph_image:
                    # Use columns to display summary and graph side by side
                    col1, col2 = st.columns([1, 1])  # Create two columns
                    with col1:
                        st.info("Summary:")
                        st.write(summary)  # Display the summary in the first column
                    with col2:
                        st.image(graph_image, caption="Generated Mermaid Graph")  # Display the diagram in the second column
                else:
                    st.warning("Failed to generate graph.")

        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(
                f"""
                <div style="background-color: #d9f7be; padding: 10px 15px; border-radius: 10px; color: #333; margin-bottom: 10px;">
                    {message["user"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with st.chat_message("ai"):
            st.markdown(
                f"""
                <div style="background-color: #e6f7ff; padding: 10px 15px; border-radius: 10px; color: #333; margin-bottom: 10px;">
                    {message["ai"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

# Main function
def main() -> None:
    # st.set_page_config(page_title=None, page_icon='🤖', layout="wide")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title('💬 Chat UI')

    # Dropdown to choose input method
    input_option = st.selectbox(
        "Choose an input method:",
        options=["Upload a Code File", "Paste a URL"]
    )

    input_data = None

    if input_option == "Upload a Code File":
        uploaded_file = st.file_uploader("Upload your code file", type=["py", "txt", "cpp", "java", "js"])
        if uploaded_file:
            input_data = uploaded_file.read().decode("utf-8")
            st.text_area("Code Preview", value=input_data, height=300, disabled=True)

    elif input_option == "Paste a URL":
        url = st.text_input("Paste the URL here:")
        if url:
            try:
                with st.spinner('Fetching document...'):
                    input_data = fetch_document(url)
                    st.text_area("Document Content", value=input_data, height=300, disabled=True)
            except Exception as e:
                st.error(f"Failed to fetch document: {str(e)}")

    diagram_type_option = st.selectbox(
            "Select the diagram type:",
            options=["Automatic Detection", "Flowchart", "Sequence Diagram", "Class Diagram", "State Diagram",
                    "Entity Relationship Diagram", "User Journey", "Gantt", "Pie Chart", "Quadrant Chart",
                    "Requirement Diagram", "Gitgraph (Git) Diagram", "C4 Diagram", "Mindmaps", "Timeline",
                    "ZenUML", "Sankey", "XY Chart", "Block Diagram", "Packet", "Kanban", "Architecture"]
        )
    
    
    if st.button('Generate Visual Summary'):
        user_question = "st.text_input('Type your question here...')"
        
        if user_question:
                handle_userinput(user_question,diagram_type_option, input_data=input_data)

if __name__ == '__main__':
    main()
