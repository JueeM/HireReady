import PyPDF2

def load_resume(pdf_path):
    text=""
    with open(pdf_path,"rb") as file:
        reader=PyPDF2.PdfReader(file)
        for page in reader.pages:
            text+=page.extract_text()
        return text
    
if __name__== "__main__":
    try:
        text = load_resume("JUEE_MAHAJAN_RESUME.pdf")
        print("\n--- SUCCESS! HERE IS A SNIPPET OF YOUR RESUME ---")
        print(text[:500])  
    except Exception as e:
        print(f"\n❌ Something went wrong: {e}")

