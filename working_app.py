import streamlit as st
import os
from transformers import AutoTokenizer
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq
from supabase import create_client, Client
import pandas as pd



# Verificar se há uma GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"


# Carregar o modelo de embeddings
transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Carregar os dados do Excel
# df = pd.read_excel('DeepSeek_Pets_Input.xlsx', sheet_name='Sheet2', header=None).dropna()
# entries = df[0].tolist()


def generate_filter_prompt(question):
    pre_prompt = '''"""
    Você é um assistente inteligente que ajuda tutores de animais de estimação a acessar e registrar informações sobre seus pets de forma clara, objetiva e útil.

    Você tem dois papéis principais:

    1. **Responder perguntas do usuário** com base exclusivamente nas informações disponíveis no contexto fornecido.
    2. **Formatar entradas de novos eventos** relatados pelo usuário em uma estrutura específica para que possam ser registradas na base de dados.

    ---

    Siga estas instruções com cuidado:

    ### Quando o usuário estiver **fazendo uma pergunta**:
    - Responda **apenas** com a palavra "Pergunta".
    - Não forneça informações adicionais ou contexto.
    - Não faça suposições ou inferências. Apenas responda "Pergunta".
    - Não formate a resposta de nenhuma maneira. Apenas "Pergunta".
    - Não use aspas ou qualquer outro tipo de formatação. Apenas "Pergunta".
    - Caso o valor for relativo a uma data, especifique o dia, mês e ano. Ex: 12/04/2025.

    ### Quando o usuário estiver **relatando um novo evento** (ex: "hoje a Mia pesou 5.9kg"):
    - Ignore o contexto anterior e **responda formatando a entrada no seguinte modelo**:

    [{Nome_do_animal} {descrição direta, completa e breve do evento com assunto e valores relevantes} em {data_do_evento}.]
        
    - Use uma linguagem natural, mas padronizada. 
    - Interprete informações úteis como sintomas, local, horário, nome do veterinário, motivo do evento, etc., para preencher corretamente a estrutura.
    - Use a data atual como referência para o evento, a menos que uma data específica seja mencionada.
    - Não ignore informações relacionadas a medicamentos, tratamentos ou cuidados recomendados.
    - Caso precise de mais informações, faça perguntas específicas para obter os dados necessários.
    - Não faça suposições ou inferências. Apenas formate a entrada de acordo com o modelo.
    - Sempre exiga o nome do animal, o evento e a data. Se não houver data, use a data atual.
    - Se o evento ou o nome do animal não for claro, pergunte ao usuário para esclarecer.
    - O nome do animal deve ser **sempre** especificado. Se não for mencionado, pergunte ao usuário para esclarecer.
    - O evento deve ser **sempre** especificado. Se não for mencionado, pergunte ao usuário para esclarecer.


    **Importante:**
    - Nunca misture as duas funções. Ou você responde "Pergunta", ou formata uma nova entrada.
    - Seja direto, útil e profissional.


    ### **Exemplo de aplicação com entrada de evento**

    **Entrada do usuário:**  
    > hoje a mia foi no veterinário (dra. Raquel) pois teve uma queda, a dra disse que ela está bem e que deve tomar 1/4 de dipirona caso tenha dor

    **Saída esperada:**  
    > [Mia teve uma consulta veterinária com a Dra. Raquel devido a uma queda em 12/04/2025. Registrou prognóstico: sem anomalias detectadas, tomar 1/4 de dipirona caso apresente sinais de dor.]

    ### Entrada do usuário:
    A seguir está a entrada do usuário. Analise cuidadosamente e aja conforme as instruções acima.


    '''
    return pre_prompt + question + '\n\n### Saída:'

def generate_prompt(entries, question):
    prompt = '''É um assistente inteligente que ajuda os donos de animais a aceder a informações sobre os seus animais de estimação de forma clara e objetiva.
                
                O seu objetivo é fornecer respostas **úteis, diretas e factuais**, baseadas exclusivamente nas informações disponíveis que lhe são fornecidas.  
                
                Siga cuidadosamente estas instruções:
                - Responda de forma **clara e concisa**, numa linguagem natural e amigável.
                - Se a informação solicitada não estiver presente no contexto fornecido, diga **"Desculpe, não consegui encontrar essa informação. ”**.
                - Concentre-se apenas no que foi perguntado. Evitar respostas demasiado longas ou desnecessárias.
                - Sempre que possível, **inclua datas, montantes e nomes de animais de estimação** com exatidão.
                - É um assistente concebido para ajudar os humanos a cuidar dos seus animais de estimação. Mantenha sempre um tom útil e respeitoso.
                
                Abaixo encontra-se a pergunta do utilizador e o contexto extraído da base de dados, se disponível.
                Utilize apenas este contexto para responder.
                '''
    # # inform today's date at the beginning of the prompt
    # prompt = f"Data de hoje: {pd.Timestamp.now().strftime('%d/%m/%Y')}\n\n" + prompt

    for i, entrie in enumerate(entries):
        prompt += f"[Entrada {i+1}]: {entrie}\n\n"
    prompt += f"Pergunta: {question}\n\nResposta: \n\nExplicação:" # acabei repetindo o answer: e explanation: porém a geração ficou boa
    return prompt

def filter_question_entrie(query):
    # results = []
    # for item in tqdm(user_questions, desc="RAG with Explanations: "):
    #     # Extrair a pergunta
    #     query = item['question']

    
    prompt = generate_filter_prompt(query)

    # Responder à pergunta com LLM
    rag_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
    )
    # Processar a resposta para separar a explicação e a resposta curta
    rag_answer = rag_response.choices[0].message.content
    # rag_answer_lower = rag_answer.lower()

    # Verifica se a resposta contém "{short answer}" e "{explanation}" ou "short answer:"
    #Porque a resposta contém {short answer} e {explanation} ou short answer:
    # {short answer} e {explanation} são placeholders que podem ser usados para separar a resposta curta da explicação.
    if "{short answer}" in rag_answer.lower() and "{explanation}" in rag_answer.lower():
        # Separar usando os placeholders {explanation} e {short answer}
        explanation = rag_answer.split("{explanation}:")[1].split("{short answer}:")[0].strip().strip("'\"")
        short_answer = rag_answer.split("{short answer}:")[1].strip().strip("'\"")
    elif "short answer:" in rag_answer.lower():
        # Separar com "short answer:"
        split_index = rag_answer.lower().find("short answer:")
        explanation = rag_answer[:split_index].strip()
        short_answer = rag_answer[split_index + len("short answer:"):].strip()
    else:
        short_answer = rag_answer.strip()
        explanation = ''
    # Armazenar os resultados
    result ={
        # "question": query,
        # "answer": item['answer'],
        "rag answer": short_answer,
        "explanation": explanation,
    }



    return result

def answer_questions_with_explanations(query):
    # results = []
    # for item in tqdm(user_questions, desc="RAG with Explanations: "):
    #     # Extrair a pergunta
    #     query = item['question']

    

    # Gerar a embedding da consulta
    query_embedding = transformer.encode([query])

    # Realizar a busca no índice (busca pelos 5 segmentos mais similares)
    top_k = 5
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Recuperar documentos mais similares
    similar_entries = []
    for i in range(top_k):
        similar_entries.append(entries[indices[0][i]])

    # Gerar o prompt para resposta
    prompt = generate_prompt(similar_entries, query)

    # Responder à pergunta com LLM
    rag_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature = temperature,
        top_p = top_p,
    )
    # Processar a resposta para separar a explicação e a resposta curta
    rag_answer = rag_response.choices[0].message.content
    # rag_answer_lower = rag_answer.lower()

    # Verifica se a resposta contém "{short answer}" e "{explanation}" ou "short answer:"
    if "{short answer}" in rag_answer.lower() and "{explanation}" in rag_answer.lower():
        # Separar usando os placeholders {explanation} e {short answer}
        explanation = rag_answer.split("{explanation}:")[1].split("{short answer}:")[0].strip().strip("'\"")
        short_answer = rag_answer.split("{short answer}:")[1].strip().strip("'\"")
    elif "short answer:" in rag_answer.lower():
        # Separar com "short answer:"
        split_index = rag_answer.lower().find("short answer:")
        explanation = rag_answer[:split_index].strip()
        short_answer = rag_answer[split_index + len("short answer:"):].strip()
    else:
        short_answer = rag_answer.strip()
        explanation = ''
    # Armazenar os resultados
    result ={
        # "question": query,
        # "answer": item['answer'],
        "rag answer": short_answer,
        "explanation": explanation,
    }

    return result

def processar_entradas():
    database = (
            supabase.table(table_name)
            .select("descricao")
            .execute()
        )

    # Extract the 'descricao' field from each dictionary in the list
    entries = np.array([item['descricao'] for item in database.data])

    embeddings = transformer.encode(entries, device=device)  # Gera as embeddings

    # Determinar a dimensionalidade das embeddings
    dimension = embeddings.shape[1]

    # Criar o índice FAISS (usando o método IndexFlatL2 para buscas baseadas na distância euclidiana)
    index = faiss.IndexFlatL2(dimension)

    # Adicionar as embeddings ao índice
    index.add(np.array(embeddings))

    return index, entries



############## APP ###########################################################

# App title
st.set_page_config(page_title="Streamlit Groq Chatbot", page_icon="🐱")

# Groq Credentials
with st.sidebar:

    st.title('🐱 Streamlit Groq Pet Chatbot', help="Caso queira inserir novas informações de credenciais, recarregue a página.")
    st.write('Projeto de chatbot usando modelos LLM via API Groq para registrar e buscar informações sobre meus pets.')

    st.write(f"Utilizando disposivo: {device}")

    st.write(f"Utilizando embedding transformer model: sentence-transformers/all-MiniLM-L6-v2")

    # Upload CSV com credenciais
    cred_file = st.file_uploader("📄 Envie um arquivo CSV com suas credenciais ou insira as informações nos campos abaixo", type=["csv"], help="O arquivo csv deve ter a primeira linha groq_api,supabase_url,supabase_key,table_name e a segunda sendo essas informações, respectivamente.")

    # Valores default (vazios)
    groq_api = ""
    supabase_url = ""
    supabase_key = ""
    table_name = ""

    # Se um arquivo CSV for enviado, ler as credenciais
    if cred_file is not None:
        try:
            creds_df = pd.read_csv(cred_file)
            if all(col in creds_df.columns for col in ["groq_api", "supabase_url", "supabase_key", "table_name"]):
                st.session_state.groq_api = creds_df.loc[0, "groq_api"]
                st.session_state.supabase_url = creds_df.loc[0, "supabase_url"]
                st.session_state.supabase_key = creds_df.loc[0, "supabase_key"]
                st.session_state.table_name = creds_df.loc[0, "table_name"]
                st.success("✅ Credenciais carregadas do CSV!")
            else:
                st.error("O CSV deve conter as colunas: groq_api, supabase_url, supabase_key, table_name.")
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {e}")

    # groq_api = st.text_input('Insira o token do Groq API:', type='password')
    # if not (len(groq_api)>0):
    #     st.warning('Por favor insira seu token do Groq API.', icon='⚠️')
    #     st.markdown("**Não tem um token de API?** Dirija-se ao [Groq](https://groq.com/) para obter um sem custo.")

    if "groq_api" not in st.session_state:
        groq_api = st.text_input('Insira o token do Groq API:', type='password')
        st.warning('Por favor insira seu token do Groq API.', icon='⚠️')
        st.markdown("**Não tem um token de API?** Dirija-se ao [Groq](https://groq.com/) para obter um sem custo.")
        if groq_api:
            st.session_state.groq_api = groq_api
    else:
        groq_api = st.session_state.groq_api

    # os.environ['GROQ_API_TOKEN'] = groq_api

    # Criar o cliente Groq
    # client = Groq(api_key=groq_api)
    client = Groq(api_key=st.session_state.groq_api)

    # supabase_url = st.text_input('Insira a URL para seu projeto no Supabase:')
    if "supabase_url" not in st.session_state:
        supabase_url = st.text_input('Insira a URL para seu projeto no Supabase:')
        if supabase_url:
            st.session_state.supabase_url = supabase_url
    else:
        supabase_url = st.session_state.supabase_url

    # supabase_key = st.text_input('Insira a chave do seu projeto no Supabase:', type='password')
    if "supabase_key" not in st.session_state:
        supabase_key = st.text_input('Insira a chave do seu projeto no Supabase:', type='password')
        if supabase_key:
            st.session_state.supabase_key = supabase_key
    else:
        supabase_key = st.session_state.supabase_key

    # table_name = st.text_input('Insira o nome da sua tabela:')
    if "table_name" not in st.session_state:
        table_name = st.text_input('Insira o nome da sua tabela do seu projeto Supabase:')
        if table_name:
            st.session_state.table_name = table_name
    else:
        table_name = st.session_state.table_name

    if not (len(supabase_url)>0 and len(supabase_key)>0 and len(table_name)>0):
        st.warning('Por favor insira a URL e chave de seu projeto Supabase.', icon='⚠️')
        st.markdown("**Não tem um projeto no Supabase?** Dirija-se ao [Supabase](https://supabase.com/) para obter um sem custo.")

    # os.environ['SUPABASE_URL'] = supabase_url
    # os.environ['SUPABASE_KEY'] = supabase_key

    # supabase: Client = create_client(supabase_url, supabase_key)
    supabase: Client = create_client(st.session_state.supabase_url, st.session_state.supabase_key)
    # database = (
    #     supabase.table(table_name)
    #     .select("descricao")
    #     .execute()
    # )


    st.subheader("Modelos e parâmetros")
    model = st.selectbox("Selecione um modelo",("llama3-70b-8192", "meta-llama/llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-qwen-32b", "gemma2-9b-it" ), key="model")
    # if model == "google-deepmind/gemma-2b-it":
    #     model = "google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.7, step=0.01, help="Aleatoriedade do resultado gerado")
    if temperature >= 1:
        st.warning('Valores superiores a 1 produzem resultados mais criativos e aleatórios, bem como uma maior probabilidade de alucinação.')
    if temperature < 0.1:
        st.warning('Valores próximos de 0 produzem resultados determinísticos. O valor inicial recomendado é 0,7')
    
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help="Percentagem máxima p dos tokens mais prováveis para a geração de resultados")

# # Extract the 'descricao' field from each dictionary in the list
# entries = np.array([item['descricao'] for item in database.data])

# embeddings = transformer.encode(entries, device=device)  # Gera as embeddings

# # Determinar a dimensionalidade das embeddings
# dimension = embeddings.shape[1]

# # Criar o índice FAISS (usando o método IndexFlatL2 para buscas baseadas na distância euclidiana)
# index = faiss.IndexFlatL2(dimension)

# # Adicionar as embeddings ao índice
# index.add(np.array(embeddings))
index, entries = processar_entradas()    
print("entries:", entries)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistente", "content": "Olá! Sou seu assistente de informações de seus Pets! Me pergunte sobre informações já inseridas na minha base de dados ou me peça para inserir novas informações!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistente", "content": "Olá! Sou seu assistente de informações de seus Pets! Me pergunte sobre informações já inseridas na minha base de dados ou me peça para inserir novas informações!"}]
    index, entries = processar_entradas()
    print("entries:", entries)

st.sidebar.button('Limpar histórico do chat', on_click=clear_chat_history)


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# # Função para registrar a entrada no banco de dados
def register_entry_in_database(entry):
    try:
        entrada_no_database = supabase.table(table_name).insert({"descricao": entry[1:-1]}).execute()
        if entrada_no_database.data:  # Verifica se há dados na resposta
            st.success("Entrada registrada com sucesso no banco de dados!")
        elif entrada_no_database.error:  # Verifica se há um erro na resposta
            st.error(f"Erro ao registrar entrada: {entrada_no_database.error['message']}")
        else:
            st.error("Erro desconhecido ao registrar entrada.")
    except Exception as e:
        st.error(f"Erro ao registrar entrada: {str(e)}")

# Function for generating model response
def generate_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistente\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistente")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()


    response = filter_question_entrie(prompt_str)

    if response['rag answer'] == 'Pergunta':
        # print("Pergunta")
        response = answer_questions_with_explanations(prompt_str)
        yield response["rag answer"]
        # print(answer)
    else:
        yield response["rag answer"]

    

# User-provided prompt
if prompt := st.chat_input(disabled=not groq_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistente
if st.session_state.messages[-1]["role"] != "assistente":
    with st.chat_message("assistente"):
        response = generate_response()

        full_response = st.write_stream(response)  

        # extract the gerenated text from the response
        if full_response.startswith("["):
            st.write("Devo registrar esta entrada na base de dados? \n\nCaso deseje editar a entrada escreva a edição desejada.")
            st.session_state.pending_entry = full_response  # Armazene a entrada pendente para registro
        else:
            message = {"role": "assistente", "content": full_response}
            st.session_state.messages.append(message)
          
    # message = {"role": "assistente", "content": full_response}
    # st.session_state.messages.append(message)

# Verifique a resposta do usuário para registrar a entrada
if "pending_entry" in st.session_state and st.session_state.pending_entry:
    user_response = st.session_state.messages[-1]["content"].strip().lower()
    if user_response == "sim":
        register_entry_in_database(st.session_state.pending_entry)
        st.session_state.pending_entry = None  # Limpe a entrada pendente
        # refresh the page to clear the input box
        # st.experimental_rerun()
    elif user_response == "não" or user_response == "nao":
        st.session_state.pending_entry = None  # Limpe a entrada pendente



# referencias:
# https://github.com/streamlit/cookbook/tree/main/recipes/replicate?ref=blog.streamlit.io
# https://blog.streamlit.io/how-to-create-an-ai-chatbot-llm-api-replicate-streamlit/
# https://github.com/LucasCoutoLima/RAG/blob/main/RAG.ipynb
