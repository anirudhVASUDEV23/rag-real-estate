# import streamlit as st
# from rag import process_urls,generate_answer


# st.title("Real Estate Research Tool")


# url1=st.sidebar.text_input("URL 1")
# url2=st.sidebar.text_input("URL 2")
# url3=st.sidebar.text_input("URL 3")

# placeholder=st.empty()


# process_url_button=st.sidebar.button("Process URL")

# if process_url_button:
#     urls=[url for url in (url1,url2,url3) if url!='']
#     if(len(urls)==0):
#         placeholder.text("Please enter at least one URL")
#     else:
#         for status in process_urls(urls):
#             placeholder.text(status)

# query=placeholder.text_input("Ask a question")
# if query:
#     try:
#         answer,sources=generate_answer(query)
#         st.header("Answer:")
#         st.write(answer)

#         if sources:
#             st.subheader("Sources:")
#             for source in sources.split("\n"):
#                 st.write(source)

#     except RuntimeError as e:
#         placeholder.text("You must process URLs first")


import streamlit as st
from rag import process_urls, generate_answer, initialize_components # Import initialize_components too

st.title("Real Estate Research Tool")

# Initialize components once at the start of the app
# This ensures LLM and vector store are ready from the beginning
initialize_components()

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty() # For displaying messages to the user

process_url_button = st.sidebar.button("Process URL")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != '']
    if not urls: # Simplified check for empty list
        placeholder.text("Please enter at least one URL")
    else:
        # --- THIS IS THE KEY CHANGE: Call process_urls directly ---
        placeholder.text("Processing URLs... This may take a moment.")
        process_urls(urls) # Call the function directly
        placeholder.text("URL processing complete! You can now ask questions.")
        # --- END OF KEY CHANGE ---

query = placeholder.text_input("Ask a question")
if query:
    try:
        # No need to call initialize_components here again, as it's done at the top
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources and sources != "No sources found (database empty).": # Only show if sources exist and not default message
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
        elif sources == "No sources found (database empty).":
            st.info("No documents were loaded into the database, so the answer is from general knowledge.")

    except RuntimeError as e:
        # This error is raised by generate_answer if vector_store is not initialized,
        # but initialize_components at the top should prevent it.
        st.error(f"An application error occurred: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
