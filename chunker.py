from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n","\n"," "]
    )
    chunks=splitter.split_text(text)
    return chunks

if __name__=="__main__":
    from loader import load_resume
    try:
        my_resume_path = r"C:\Users\Admin\Downloads\JUEE_MAHAJAN_RESUME.pdf"
        
        text = load_resume(my_resume_path)
        chunks = chunk_text(text)
        
        print(f"\n Total chunks: {len(chunks)}")
        print(f"--- First chunk: ---\n{chunks[0]}")
        
        if len(chunks) > 1:
            print(f"\n--- Second chunk: ---\n{chunks[1]}")
            
    except Exception as e:
        print(f"\n Something went wrong: {e}")