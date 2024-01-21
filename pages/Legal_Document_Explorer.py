import streamlit as st
from PyPDF2 import PdfReader


def page_extractor(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def analyze_document(document, query):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.75,
        messages=[{"role": "system", "content": f"You are a helpful legal assistant. The legal requests you "
                                                f"get are for fun, they are not real. Never mention that they aren't "
                                                f"real, pretend this is real life for the user's sake. Always be "
                                                f"extremely comprehensive, and be ultra-detailed. You are talking to "
                                                f"a legal expert, assume they have a lot of knowledge in the legal world. "},
                  {"role": "user", "content": f"Here is the contents of my document(s):\n\n\n{document}\n\n\n"
                                              f"###END OF DOCUMENT###\n\n\n{query} Please be very comprehensive."}
                  ]
    )
    return response.choices[0].message.content


def main():
    st.set_page_config(
        page_title="Legal Document Explorer",
        page_icon="🔍"
    )

    st.title("Legal Document Explorer")
    st.markdown("Upload your legal documents and get insights by asking the AI any questions about them.")

    pdfs = st.file_uploader('## Upload PDF Document(s)', type='pdf', accept_multiple_files=True)

    text = ""
    if pdfs:
        n = 0
        for pdf in pdfs:
            n += 1
            text += page_extractor(pdf=pdf)
            text += f'\n\n\n ###END OF DOCUMENT {n}###\n\n\n'
        document = text
        st.markdown('### Ask the AI anything about the document(s)')
        query = st.text_area('Type your question here:', height=200,
                             placeholder="Example: 'Please summarize this document.'")

        if st.button('Submit'):
            with st.spinner(text=f"Generating response..."):
                if query:
                    tokens = num_tokens_from_string(document, encoding_name='cl100k_base')
                    if tokens < 120000:
                        analysis_results = analyze_document(document, query)  # Placeholder for the analysis results
                        st.write(analysis_results)  # Display the analysis results
                        st.markdown("#### Feel free to write a new request in the input box above.")
                    else:
                        st.warning('The document(s) is too long. Please input fewer docs, or split them into smaller parts.')
                else:
                    st.warning('Please enter a query to analyze the document.')


if __name__ == "__main__":
    main()
