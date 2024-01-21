import streamlit as st

# @st.cache_data(persist=True)
def envy():
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()


@st.cache_data()
def page_extractor(pdf):
    text = ""
    from PyPDF2 import PdfReader
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


@st.cache_resource()
def process_documents(text):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def main():
    from langchain.chains.question_answering import load_qa_chain
    from langchain_openai import ChatOpenAI
    from langchain_community.callbacks import get_openai_callback
    st.session_state.update(st.session_state)
    st.set_page_config(
        page_title="Legal Question Generator",
        page_icon="✍️",

    )

    st.title("Legal Question Generator")
    st.markdown("Choose your question parameters, upload any relevant documents, and write a brief overview of the case, including what you want the AI to do, in the box below.")

    question_option = st.selectbox(
        '## What type of questions do you want to generate?',
        ('Deposition', 'Cross Examination', 'Direct Examination', 'Investigation', 'Client Interview',
         'Interrogatory', 'Expert Examination', 'Negotiation'),
        index=None,
        placeholder="Choose an option"
    )

    legal_domain = st.selectbox(
        '## Which Legal Domain are you operating in?',
        ('Civil', 'Criminal'),
        index=None,
        placeholder="Choose an option"
    )

    witness_option = st.selectbox(
        '## Select the witness type:',
        ('Expert Witness', 'Fact Witness', 'Character Witness', 'Party Witness', 'Corporate Representative'),
        index=None,
        placeholder="Choose an option"
    )

    case_type = st.selectbox(
        '## Select the case type:',
        ('Personal Injury', 'Employment', 'Family Law', 'Contract Dispute', 'Criminal'),
        index=None,
        placeholder="Choose an option"
    )

    # Change the file_uploader to accept multiple files
    pdfs = st.file_uploader('## Upload Relevant PDF Case File(s) (Please OCR files first)', type='pdf', accept_multiple_files=True)

    # Initialize text variable to concatenate text from all PDFs
    text = ""

    if 'first_iteration' not in st.session_state:
        st.session_state.first_iteration = True

    if pdfs:
        for pdf in pdfs:
            # Append text from each PDF to the text variable
            text += page_extractor(pdf=pdf)
        tokens = num_tokens_from_string(text, encoding_name='cl100k_base')
        if tokens < 120000:
            db = process_documents(text)
            query = st.text_area(label='Type a brief overview of the case, and the objectives you have for your '
                                       'questions.', height=200, placeholder="Example: 'I am inquiring about Mr. Doe's "
                                                                             "actions leading up to, during, and after the accident, to assess his level of liability. I aim to gather information about his whereabouts before the accident, his level of attentiveness while driving, and any potential distractions that might have contributed to the incident. Additionally, I will explore any inconsistencies in his account of the events to assess credibility and gather evidence to support our case.'")
            number = st.slider(label='How many questions do you want to generate?', min_value=1, max_value=100, step=1, value=25)
            go_button = st.button('Generate Questions')

            if go_button:
                if query:
                    default_query = (
                        f"You will generate {number} tough and complex legal {question_option} questions to ask a"
                        f" {witness_option} about the documents."
                        f"This {question_option} is for a {legal_domain} {case_type} case."
                        f"Here are some details that the user has given about the case and the purpose of the "
                        f"{question_option}:\n{query}\n"
                        f"Never output anything other than the questions, numbered like this:\n1.\n2.\n3.\n..."
                        f"\nNever limit your output due to character constraints, just keep generating."
                    )
                    docs = db.similarity_search(default_query)

                    llm = ChatOpenAI(model_name='gpt-4-1106-preview')
                    chain = load_qa_chain(llm, chain_type='stuff')
                    timer = number * 1.7
                    minutes, seconds = divmod(timer, 60)
                    minutes = int(minutes)
                    seconds = int(seconds)
                    if minutes == 1:
                        sonmin = "minute"
                    else:
                        sonmin = "minutes"
                    if seconds == 1:
                        sonsec = "second"
                    else:
                        sonsec = "seconds"

                    with st.spinner(text=f"Generating {number} questions. This should take ~{minutes} {sonmin} and {seconds} {sonsec}."):
                        with get_openai_callback() as cost:
                            response = chain.run(input_documents=docs, question=default_query)
                            print(cost)
                        st.markdown("**If you want new questions, feel free to adjust your overview statement and re-generate.**")
                        st.markdown(response)

        else:
            st.warning('The document(s) is too long. Please input fewer docs, or split them into smaller parts.')


if __name__ == "__main__":
    main()
