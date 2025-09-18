import streamlit as st
from datetime import datetime
import streamlit_authenticator as stauth
import bcrypt
import pandas as pd
import numpy as np
from io import BytesIO
import dropbox
from openai import OpenAI
import altair as alt
import plotly.express as px
import faiss
import pickle
import time
from sqlalchemy import create_engine
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import git
import os
import io
import xlrd

# -----------------------------------------------------------------------------
# 0. Settings & Secrets
# -----------------------------------------------------------------------------
st.set_page_config(page_title="TaxbaseAI - Sua AI Contábil",
                   page_icon="assets/favIcon.png",
                   layout="wide"
                )

dbx_cfg      = st.secrets["dropbox"]
dbx          = dropbox.Dropbox(
    app_key=dbx_cfg["app_key"],
    app_secret=dbx_cfg["app_secret"],
    oauth2_refresh_token=dbx_cfg["refresh_token"]
)
BASE_PATH    = dbx_cfg["base_path"].rstrip("/")

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# -----------------------------------------------------------------------------
# 0.1 Credentials & Authentication (via BD)
# -----------------------------------------------------------------------------
# Conexão com banco local. Use caminho absoluto se preferir garantir:
# engine = create_engine("sqlite:///C:/caminho/para/usuarios.db")
engine = create_engine("sqlite:///usuarios.db")

def load_users_from_db() -> dict:
    try:
        df = pd.read_sql("SELECT username, name, password, empresa, role FROM usuarios", engine)
    except Exception as e:
        st.error(f"Erro ao ler usuários do banco: {e}")
        return {}
    users = {}
    for _, row in df.iterrows():
        users[row["username"]] = {
            "name": row["name"],
            "password": row["password"],  # hash bcrypt armazenado no BD
            "empresa": row["empresa"],
            "role": row["role"],
        }
    return users

def add_user_to_db(username: str, name: str, password: str, empresa: str, role: str):
    """Adiciona um novo usuário ao banco de dados com senha hasheada."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = pd.DataFrame([{
        "username": username,
        "name": name,
        "password": hashed_password,
        "empresa": empresa,
        "role": role
    }])
    try:
        new_user.to_sql("usuarios", engine, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao adicionar usuário: {e}")
        return False

def delete_user_from_db(username_to_delete: str) -> bool:
    """Deleta um usuário do banco de dados."""
    if not username_to_delete:
        return False
    try:
        # Usar a conexão do engine para executar o comando
        with engine.connect() as connection:
            # Usar 'text' para evitar SQL Injection
            from sqlalchemy import text
            stmt = text("DELETE FROM usuarios WHERE username = :username")
            connection.execute(stmt, {"username": username_to_delete})
            # Importante: para que a deleção seja efetivada
            connection.commit()
        return True
    except Exception as e:
        st.error(f"Erro ao deletar usuário: {e}")
        return False

def get_all_users() -> pd.DataFrame:
    """Carrega todos os usuários para exibição."""
    try:
        return pd.read_sql("SELECT username, name, empresa, role FROM usuarios", engine)
    except:
        return pd.DataFrame() # Retorna DF vazio se a tabela não existir

def build_credentials(users: dict) -> dict:
    return {
        "usernames": {
            user: {"name": info["name"], "password": info["password"]}
            for user, info in users.items()
        }
    }

USERS = load_users_from_db()
credentials = build_credentials(USERS)

cfg = st.secrets["auth"]
authenticator = stauth.Authenticate(
    credentials,
    cfg["cookie_name"],
    cfg["key"],
    cfg["expiry_days"],
)

# Nova assinatura com 'fields' (biblioteca atual)
authenticator.login(
    location="main",
    fields={
        "Form name": "Login",
        "Username": "Usuário",
        "Password": "Senha",
        "Login": "Entrar"
    }
)

# após exibir o formulário, leia os valores do session_state
authentication_status = st.session_state.get("authentication_status")
name                  = st.session_state.get("name")
username              = st.session_state.get("username")

# Conexão com o novo banco de dados de empresas
engine_empresas = create_engine("sqlite:///empresas.db")

def load_companies_from_db() -> list[str]:
    """Carrega a lista de nomes de empresas do banco de dados."""
    try:
        df = pd.read_sql("SELECT name FROM empresas", engine_empresas)
        return df["name"].tolist()
    except Exception:
        # Se a tabela não existir, cria e retorna a lista inicial
        initial_companies = pd.DataFrame({"name": ["CICLOMADE", "JJMAX", "SAUDEFORMA"]})
        initial_companies.to_sql("empresas", engine_empresas, index=False)
        return initial_companies["name"].tolist()

def add_company_to_db(company_name: str) -> bool:
    """Adiciona uma nova empresa ao banco de dados."""
    company_name = company_name.upper().strip() # Padroniza o nome
    current_companies = [c.upper() for c in load_companies_from_db()]
    if company_name in current_companies:
        st.warning(f"Empresa '{company_name}' já existe.")
        return False
    
    new_company = pd.DataFrame([{"name": company_name}])
    try:
        new_company.to_sql("empresas", engine_empresas, if_exists="append", index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao adicionar empresa: {e}")
        return False

# Carregue a lista de empresas no início do script
available_companies = load_companies_from_db()

def send_invitation_email_sendgrid(recipient_email: str, temp_password: str):
    """Envia um e-mail de convite usando a API do SendGrid."""
    sg_cfg = st.secrets["sendgrid"]
    
    # Cria a mensagem usando a classe Mail do SendGrid
    message = Mail(
        from_email=sg_cfg["sender_email"],
        to_emails=recipient_email,
        subject='Você foi convidado para a TaxbaseAI!',
        html_content=f"""
        <html>
        <body>
            <h2>Bem-vindo(a) à TaxbaseAI!</h2>
            <p>Sua conta foi criada com sucesso. Use as credenciais abaixo para acessar a plataforma:</p>
            <ul>
                <li><strong>Usuário:</strong> {recipient_email}</li>
                <li><strong>Senha Provisória:</strong> {temp_password}</li>
            </ul>
            <p>Recomendamos que você altere sua senha no primeiro acesso.</p>
            <a href="https://taxbase-ai.streamlit.app/" style="background-color: #4CAF50; color: white; padding: 14px 25px; text-align: center; text-decoration: none; display: inline-block;">Acessar a Plataforma</a>
            <br><br>
            <p>Atenciosamente,<br>Equipe TaxbaseAI</p>
        </body>
        </html>
        """
    )

    # Tenta enviar o e-mail
    try:
        sg = SendGridAPIClient(sg_cfg["api_key"])
        response = sg.send(message)
        # O SendGrid retorna um status code 22 Accepted em caso de sucesso
        if response.status_code == 202:
            st.info(f"E-mail de convite enviado para {recipient_email}.")
        else:
            st.error(f"Erro do SendGrid ao enviar e-mail: Status {response.status_code}")
            st.error(response.body)
    except Exception as e:
        st.error(f"Falha ao tentar enviar e-mail via SendGrid: {e}")

def git_auto_commit(commit_message: str):
    """
    Verifica por mudanças nos arquivos .db, e faz add, commit, e push se houver.
    """
    try:
        repo_path = os.getcwd() # Pega o diretório atual do script
        repo = git.Repo(repo_path)

        # Verifica se o repositório tem alterações não rastreadas ou modificadas
        if not repo.is_dirty(untracked_files=True):
            # st.info("Nenhuma alteração no banco de dados para sincronizar.")
            return

        st.write("Detectei alterações no banco de dados. Sincronizando com o GitHub...")

        # Adiciona os arquivos de banco de dados
        db_files_to_add = [f for f in ["usuarios.db", "empresas.db"] if os.path.exists(f)]
        if not db_files_to_add:
            st.warning("Nenhum arquivo de banco de dados encontrado para o commit.")
            return

        repo.index.add(db_files_to_add)

        # Faz o commit
        repo.index.commit(commit_message)
        st.write(f"✓ Commit realizado: '{commit_message}'")

        # Configura a URL remota com o token para autenticação
        github_cfg = st.secrets["github"]
        remote_url = f"https://{github_cfg['username']}:{github_cfg['token']}@github.com/{github_cfg['repo_name']}.git"

        # Faz o push
        origin = repo.remote(name='origin')
        origin.set_url(remote_url) # Define a URL com o token temporariamente
        origin.push()

        st.success("🚀 Alterações sincronizadas com sucesso no GitHub!")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a sincronização com o Git: {e}")

# -----------------------------------------------------------------------------
# 1. FAISS Embeddings
# -----------------------------------------------------------------------------
EMBED_INDEX_PATH = "embeddings.index"
META_PATH        = "embeddings_meta.pkl"
EMB_DIM          = 1536

def build_or_load_index():
    try:
        index = faiss.read_index(EMBED_INDEX_PATH)
        meta  = pickle.load(open(META_PATH, "rb"))
    except:
        index = faiss.IndexFlatL2(EMB_DIM)
        meta  = []
    return index, meta

def persist_index(index, meta):
    faiss.write_index(index, EMBED_INDEX_PATH)
    pickle.dump(meta, open(META_PATH, "wb"))

def semantic_search(query: str, index, meta, top_k=5):
    q_emb = client.embeddings.create(model="text-embedding-ada-002", input=[query]).data[0].embedding
    D, I  = index.search(np.array([q_emb], dtype="float32"), top_k)
    return [meta[i] for i in I[0] if 0 <= i < len(meta)]

def upsert_embedding(question: str, answer: str, index, meta):
    emb = client.embeddings.create(
        model="text-embedding-ada-002", input=[question+" ||| "+answer]
    ).data[0].embedding
    index.add(np.array([emb], dtype="float32"))
    meta.append({"q": question, "a": answer})
    persist_index(index, meta)

index, meta = build_or_load_index()

# -----------------------------------------------------------------------------
# 2. Ingestão & Normalização & Mapeamento
# -----------------------------------------------------------------------------
COMMON_COLUMNS = {
    "nome_empresa":"company", "descrição":"account", "descricao":"account",
    "valor":"amount", "saldo_atual":"amount"
}

def process_accounting_csv(uploaded_file, company_name: str) -> pd.DataFrame | None:
    """
    Processa arquivos contábeis que TÊM CABEÇALHO NA PRIMEIRA LINHA,
    usando o motor correto para cada tipo de Excel.
    """
    try:
        file_name = uploaded_file.name
        df = None
        
        if file_name.endswith('.csv'):
            content = uploaded_file.getvalue().decode('latin-1')
            df = pd.read_csv(io.StringIO(content))
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')

        if df is None:
            st.error("Formato de arquivo não suportado.")
            return None

        # --- AGORA, TRABALHAMOS COM AS COLUNAS LIDAS DIRETAMENTE DO ARQUIVO ---

        # 1. Validação: Verificamos se as colunas que você listou ('nome_cta', 'saldoatu') existem.
        required_cols = ["nome_cta", "saldoatu"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"O arquivo não parece ser um balancete válido. Colunas esperadas: {required_cols}. Colunas encontradas: {df.columns.tolist()}")
            return None

        # 2. Mapeamento: Renomeamos 'nome_cta' e 'saldoatu' para o padrão do sistema.
        df = df.rename(columns={"nome_cta": "account", "saldoatu": "amount"})

        # 3. Limpeza (Opcional, mas mantido por segurança)
        # df = df[df['account'] != 'Total Geral']

        # 4. Padronização
        df = df.dropna(subset=['amount'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # 5. Adição de Metadados
        df['company'] = company_name

        final_df = df[["company", "account", "amount"]]
        
        return final_df

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Detalhe: {e}")
        return None

def upload_file_to_dropbox(file_bytes: bytes, dropbox_path: str) -> bool:
    """
    Faz o upload de um conteúdo em bytes para um caminho específico no Dropbox,
    sobrescrevendo se o arquivo já existir.
    """
    try:
        dbx.files_upload(
            file_bytes,
            dropbox_path,
            mode=dropbox.files.WriteMode('overwrite')
        )
        return True
    except Exception as e:
        st.error(f"Erro ao fazer upload para o Dropbox: {e}")
        return False

@st.cache_data
def load_csv_from_dropbox(filename: str, expected_cols: list[str]) -> pd.DataFrame | None:
    path = f"{BASE_PATH}/{filename}"
    try:
        _, res = dbx.files_download(path=path)
    except dropbox.exceptions.ApiError:
        st.warning(f"Arquivo não encontrado: {filename}")
        return None
    df = pd.read_csv(BytesIO(res.content))
    if missing := set(expected_cols) - set(df.columns):
        st.error(f"Colunas faltando em {filename}: {missing}")
        return None
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COMMON_COLUMNS).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def add_metadata(df, stmt, ref_date, cid):
    df["statement"]  = stmt
    df["ref_date"]   = pd.to_datetime(ref_date)
    df["company_id"] = cid
    return df

def apply_account_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df["account_std"] = df["account"]
    return df

def clean_data(df):
    return df.dropna(subset=["amount"]).query("amount != 0").drop_duplicates(subset=["account_std", "amount"])

@st.cache_data
def load_monthly_csv_from_dropbox(prefix_month: str, company_id: str, expected_cols: list[str]) -> pd.DataFrame | None:
    try:
        entries = dbx.files_list_folder(BASE_PATH).entries
    except dropbox.exceptions.ApiError:
        st.warning(f"Erro listando {BASE_PATH}")
        return None
    
    pattern_prefix = prefix_month
    suffix = f"_{company_id}.csv"
    candidates = [e.name for e in entries if e.name.startswith(pattern_prefix) and e.name.endswith(suffix)]

    if not candidates:
        st.warning(f"Nenhum arquivo começando com '{pattern_prefix}' e terminando em '{suffix}'")
        return None
    
    latest = sorted(candidates)[-1]
    return load_csv_from_dropbox(latest, expected_cols)

def load_and_clean(company_id: str, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    dre_raw = load_monthly_csv_from_dropbox(prefix_month=f"DRE_{date_str}", company_id=company_id,
                                    expected_cols=["nome_empresa","descrição","valor"])
    bal_raw = load_monthly_csv_from_dropbox(prefix_month=f"BALANCO_{date_str}", company_id=company_id,
                                    expected_cols=["nome_empresa","descrição","saldo_atual"])
    if dre_raw is None or bal_raw is None:
        return None, None

    # padroniza, mapeia, adiciona metadata e limpa
    dre_df = standardize_columns(dre_raw)
    dre_df = apply_account_mapping(dre_df)
    dre_df = add_metadata(dre_df, "income_statement", date_str, company_id)
    dre = clean_data(dre_df)

    #Trata despesas operacionais sempre com valor positivo
    dre.loc[dre["account_std"] == "operating_expenses", "amount"] = dre.loc[dre["account_std"] == "operating_expenses", "amount"].abs()

    # padroniza, mapeia, adiciona metadata e limpa
    bal_df = standardize_columns(bal_raw)
    bal_df = apply_account_mapping(bal_df)
    bal_df = add_metadata(bal_df, "balance_sheet", date_str, company_id)
    bal = clean_data(bal_df)
    return dre, bal

# -----------------------------------------------------------------------------
# 3. Cálculo de Indicadores
# -----------------------------------------------------------------------------
def compute_indicators(dre: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    # --- DRE: sumários por padrão, em UPPER para regex mais seguro
    d = dre.copy()
    d["DESC"] = d["account"].str.upper()

    def sum_dre(regex: str, absolute: bool = False) -> float:
        """Soma todos os amounts cujas DESCRIÇÕES batem o regex."""
        vals = d.loc[d["DESC"].str.contains(regex, regex=True), "amount"]
        return vals.abs().sum() if absolute else vals.sum()

    # 1) Receita Líquida (linha exata)
    receita_liq = sum_dre(r"^RECEITA LÍQUIDA$")

    # 2) Lucro Bruto: se houver linha, usamos ela; senão receita - custos
    lucro_bruto = sum_dre(r"^LUCRO BRUTO$") or (
        receita_liq - sum_dre(r"CUSTOS DOS PRODUTOS VENDIDOS|CUSTOS DE MERCADORIAS", absolute=True)
    )

    # 3) Depreciações (tudo que contenha "DEPRECIA")
    deprec = sum_dre(r"DEPRECIA", absolute=True)

    # 4) Despesas Operacionais = soma de todas as linhas que começam com "DESPESAS"
    total_desp = sum_dre(r"^(-\s*)?DESPESAS", absolute=True)
    despesas_op = total_desp - deprec

    # 5) EBITDA: prefira RESULTADO OPERACIONAL + depreciações
    resultado_oper = sum_dre(r"^RESULTADO OPERACIONAL$")
    if resultado_oper:
        ebitda = resultado_oper + deprec
    else:
        # fallback para casos sem "RESULTADO OPERACIONAL"
        ebitda = lucro_bruto - despesas_op

    # 6) Lucro Líquido ou Prejuízo
    lucro_liq = sum_dre(r"^LUCRO LÍQUIDO DO EXERCÍCIO$|^LUCRO LÍQUIDO$|^PREJUÍZO DO EXERCÍCIO$")

    # --- balanço patrimonial ---
    b = bal.copy()
    b["DESC"] = b["account"].str.upper()
    def sum_bal(regex: str, absolute: bool = False) -> float:
        vals = b.loc[b["DESC"].str.contains(regex, regex=True), "amount"]
        return vals.abs().sum() if absolute else vals.sum()

    ativo_circ = sum_bal(r"^ATIVO CIRCULANTE$")
    pass_circ  = sum_bal(r"^PASSIVO CIRCULANTE$")
    estoque    = sum_bal(r"^ESTOQUE$")

    liquidez_corrente = ativo_circ / pass_circ if pass_circ else None
    liquidez_seca     = (ativo_circ - estoque) / pass_circ if pass_circ else None

    # Ativos totais: tenta “ATIVO”: senão circulante + permanente
    total_ativo = sum_bal(r"^ATIVO$") or (
        sum_bal(r"^ATIVO CIRCULANTE$") +
        sum_bal(r"ATIVO PERMANENTE|ATIVO NÃO-CIRCULANTE")
    )
    # passivos totais = circulante + não-circulante
    pass_circ  = sum_bal(r"^PASSIVO CIRCULANTE$",   absolute=True)
    pass_ncirc = sum_bal(r"PASSIVO NÃO-CIRCULANTE", absolute=True)
    total_pass = pass_circ + pass_ncirc
    endividamento = total_pass / total_ativo if total_ativo else None

    # primeiro tenta achar patrimônio líquido expresso no CSV
    patr_liq = sum_bal(r"^PATRIMÔNIO LÍQUIDO$")
    if patr_liq == 0:
        # fallback: usar lucros ou prejuízos acumulados
        patr_liq = sum_bal(r"LUCROS OU PREJUÍZOS ACUMULADOS", absolute=True)
    roa  = lucro_liq / total_ativo if total_ativo else None
    roe  = lucro_liq / patr_liq if patr_liq else None

    return pd.DataFrame({
        "Indicador": [
            "Lucro Bruto", "EBITDA", "Lucro Líquido",
            "Liquidez Corrente", "Liquidez Seca", "Endividamento",
            "ROA", "ROE"
        ],
        "Valor": [
            lucro_bruto, ebitda, lucro_liq,
            liquidez_corrente, liquidez_seca, endividamento,
            roa, roe
        ]
    })
# -----------------------------------------------------------------------------
# 3.1 Geração de perguntas de acompanhamento (respostas encadeadas)
# -----------------------------------------------------------------------------
def generate_followups(user_prompt: str, assistant_answer: str, company: str, date_str: str) -> list[str]:
    """Gera 2 perguntas curtas e úteis para continuar a conversa."""
    try:
        follow = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"Gere 2 perguntas curtas, objetivas, úteis, sobre finanças/contabilidade com base no diálogo. Responda apenas com uma lista separada por linhas, sem prefixos."},
                {"role":"user","content":f"Empresa: {company} | Data: {date_str}\nPergunta do usuário: {user_prompt}\nResposta da IA: {assistant_answer}"}
            ],
            temperature=0.3,
            max_tokens=120
        ).choices[0].message.content.strip()
        # Quebra por linhas, remove vazios e limita a 2
        suggestions = [s.strip(" -•\t") for s in follow.splitlines() if s.strip()]
        return suggestions[:2] if suggestions else []
    except Exception:
        # fallback
        return [
            "Quer ver a evolução desses indicadores versus o período anterior?",
            "Deseja que eu detalhe a composição das despesas operacionais?"
        ]
    
# -----------------------------------------------------------------------------
# 3.2 Personalidade & Contexto Contínuo
# -----------------------------------------------------------------------------
TONE_SYSTEM = (
    "Você é um assistente contábil com voz inteligente e próxima. "
    "Estilo: claro, direto, sem jargões desnecessários, e sempre útil. "
    "Use linguagem simples, destaque números importantes com contexto de negócio."
)

def brief_history(messages: list[dict], limit: int = 6, max_chars: int = 900) -> str:
    if not messages:
        return ""
    tail = messages[-limit:]
    lines = []
    for m in tail:
        role = "Usuário" if m["role"] == "user" else "IA"
        txt = m["content"].strip()
        if len(txt) > 300:
            txt = txt[:300] + "..."
        lines.append(f"{role}: {txt}")
    return "\n".join(lines)[:max_chars]

def initial_greeting(company: str, date_str: str) -> str:
    return (
        f"Oi — eu sou a sua AI contábil. Vamos olhar a {company} em {date_str}? "
        "Posso analisar margens, liquidez, variações relevantes e sugerir próximos passos. "
        "Comece pedindo um raio-x rápido ou perguntando por um indicador específico."
    )

# -----------------------------------------------------------------------------
# 4. UI & Navigation
# -----------------------------------------------------------------------------
if authentication_status:
    if username not in USERS:
        st.error("Usuário autenticado não encontrado no banco de dados.")
        st.stop()

    user_info = USERS[username]
    role      = user_info["role"]
    empresa   = user_info["empresa"]

    with st.sidebar:
        st.image("assets/taxbaseAI_logo.png", width=250)
        st.divider()

    authenticator.logout("Sair", "sidebar")
    st.sidebar.success(f"Conectado como {user_info['name']} ({role})")

    available_companies = load_companies_from_db()
    if role == "admin":
        session_companies = st.sidebar.multiselect(
            "Selecione empresas", available_companies, default=available_companies
        )
    else:
        session_companies = [empresa] if empresa else []

    # Filtrar apenas empresas válidas
    session_companies = [c for c in session_companies if c in available_companies]
    if not session_companies:
        st.sidebar.error("Selecione ao menos uma empresa válida.")

    session_date = st.sidebar.date_input("Mês de Referência", value=pd.to_datetime("2024-12-31"))
    date_str = session_date.strftime("%Y-%m")

    company_for_metrics = st.sidebar.selectbox("Empresa para Métricas", session_companies) if session_companies else None

    # Carregar dados com tratamento de erros
    all_dre, all_bal = [], []
    for comp in session_companies:
        dre, bal = load_and_clean(comp, date_str)
        if dre is None or bal is None:
            st.warning(f"Pulando {comp}: dados não encontrados.")
            continue
        all_dre.append(dre)
        all_bal.append(bal)

    if all_dre and all_bal:
        df_all = pd.concat(all_dre + all_bal, ignore_index=True)
    else:
        df_all = pd.DataFrame()  # vazio

    page = st.sidebar.radio("📊 Navegação", ["Visão Geral", "Dashboards", "TaxbaseAI"])

    if role == "admin":
        if st.sidebar.button("Painel do Administrador"):
            st.session_state.page = "Admin"

    active_page = st.session_state.get("page", page)

    if active_page == "Admin" and role == "admin":
        st.header("🔑 Painel do Administrador")
        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["Gerenciar Usuários", "Gerenciar Empresas", "📤 Upload de Relatórios"])

        with admin_tab1:
            st.subheader("➕ Criar Novo Usuário")
            with st.form("new_user_form", clear_on_submit=True):
                new_email = st.text_input("E-mail do Usuário (para login)")
                new_username = st.text_input("Usuário (para login)")
                new_name = st.text_input("Nome Completo")
                new_password = st.text_input("Senha Provisória", type="password")
                # Assumindo que a lista de empresas virá de um BD ou está definida
                # (Vamos implementar isso na seção 3)
                available_companies = load_companies_from_db() # Placeholder
                assigned_company = st.selectbox("Empresa de Acesso", options=available_companies)
                assigned_role = st.selectbox("Role", options=["user", "admin"])
        
                submitted = st.form_submit_button("Criar Usuário")

                if submitted:
                    if new_email and new_name and new_password:
                        if add_user_to_db(username=new_email, name=new_name, password=new_password, empresa=assigned_company, role=assigned_role):
                            st.success(f"Usuário '{new_name}' criado com sucesso!")

                            send_invitation_email_sendgrid(recipient_email=new_email, temp_password=new_password)
                            git_auto_commit(commit_message=f"feat: Adiciona novo usuário '{new_name}'")
                        else:
                            st.error("Não foi possível criar o usuário.")
                    else:
                        st.warning("Por favor, preencha todos os campos.")

            st.divider()
            st.subheader("👥 Usuários Existentes")
            st.dataframe(get_all_users(), use_container_width=True)
            st.divider()

            st.subheader("🗑️ Deletar Usuário")
            # Pega a lista de todos os usernames, exceto o do admin logado
            users_df = get_all_users()
            # Garante que a coluna 'username' existe antes de tentar filtrar
            if not users_df.empty and 'username' in users_df.columns:
                users_list = users_df["username"].tolist()
                # Impede que o admin se delete
                current_admin_username = st.session_state.get("username")
                if current_admin_username in users_list:
                    users_list.remove(current_admin_username)

                user_to_delete = st.selectbox(
                    "Selecione o usuário para deletar",
                    options=users_list,
                    index=None,
                    placeholder="Escolha um usuário..."
                )

                if user_to_delete:
                    st.warning(f"**Atenção:** Você está prestes a deletar o usuário **{user_to_delete}**. Esta ação é irreversível.")
                    if st.button(f"Confirmar Deleção de {user_to_delete}"):
                        if delete_user_from_db(user_to_delete):
                            st.success(f"Usuário '{user_to_delete}' deletado com sucesso!")
                            git_auto_commit(f"refactor: Deleta usuário '{user_to_delete}'")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Falha ao deletar o usuário.")
                else:
                    st.info("Nenhum outro usuário para deletar.")

        with admin_tab2:
            st.subheader("🏢 Adicionar Nova Empresa")
            with st.form("new_company_form", clear_on_submit=True):
                new_company_name = st.text_input("Nome da Nova Empresa")
                submitted_company = st.form_submit_button("Adicionar Empresa")

                if submitted_company and new_company_name:
                    if add_company_to_db(new_company_name):
                        st.success(f"Empresa '{new_company_name.upper()}' adicionada com sucesso!")
                        git_auto_commit(commit_message=f"feat: Adiciona nova empresa '{new_company_name.upper()}'")
                        st.info("A página será recarregada para atualizar as listas.")
                        time.sleep(3) # Pausa para o usuário ler a mensagem
                        st.rerun() # Recarrega a página para que a nova empresa apareça nas seleções
                    else:
                        st.error("Falha ao adicionar a empresa.")

            st.divider()
            st.subheader("📋 Empresas Cadastradas")
            st.dataframe(pd.DataFrame({"Nome": load_companies_from_db()}), use_container_width=True)

        with admin_tab3:
            st.subheader("Upload de Relatórios Mensais (DRE / Balanço)")
            st.info("Esta área está adaptada para ler os arquivos de balancete (com cabeçalho e rodapé) extraídos do sistema contábil.")

            # Usar um formulário garante que todos os dados sejam enviados de uma vez
            with st.form("upload_form_v3", clear_on_submit=True):
                company_to_upload = st.selectbox(
                    "Para qual empresa é este relatório?",
                    options=load_companies_from_db(),
                    index=None,
                    placeholder="Selecione a empresa"
                )

                # --- INÍCIO DA CORREÇÃO DE DATA ---
                # Criamos listas para os seletores de mês e ano
                current_year = datetime.now().year
        
                col1, col2 = st.columns(2)
                with col1:
                    selected_month = st.selectbox(
                        "Mês de Referência",
                        options=range(1, 13),
                        format_func=lambda month: f"{month:02d}", # Formata para "01", "02", etc.
                        index=None,
                        placeholder="Selecione o Mês"
                    )
                with col2:
                    selected_year = st.selectbox(
                        "Ano de Referência",
                        options=range(current_year - 5, current_year + 1), # Últimos 5 anos + ano atual
                        index=None,
                        placeholder="Selecione o Ano"
                    )

                report_type = st.selectbox(
                    "Qual o tipo de relatório?",
                    options=["DRE", "BALANCO"],
                    index=None,
                    placeholder="Selecione o tipo"
                )
                # --- FIM DA CORREÇÃO DE DATA ---

                uploaded_file = st.file_uploader(
                    "Arraste ou selecione o arquivo (CSV, XLS ou XLSX)",
                    type=["csv", "xls", "xlsx"]
                )

                # O botão de envio está DENTRO do 'with st.form'
                submitted = st.form_submit_button("Processar e Enviar Arquivo")

            # --- Lógica do Backend ---
            if submitted:
                if not all([company_to_upload, selected_month, selected_year, uploaded_file]):
                    st.warning("Por favor, preencha todos os campos e anexe um arquivo.")
                else:
                    # Reconstrói a data no formato que o sistema precisa
                    date_str = f"{selected_year}-{selected_month:02d}"

                    cleaned_df = process_accounting_csv(uploaded_file, company_to_upload)

                    # A função retorna um DataFrame se tudo deu certo, ou None se deu erro
                    if cleaned_df is not None:
                        new_filename = f"{report_type}_{date_str}_{company_to_upload}.csv"
                        full_dropbox_path = f"{BASE_PATH}/{new_filename}"

                        st.info(f"Arquivo processado com sucesso. Enviando para o sistema como '{new_filename}'...")

                        # 3. Converte o DataFrame limpo de volta para um CSV em memória
                        # Isso garante que o arquivo salvo no Dropbox seja simples e padronizado
                        csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')

                        # 4. Faz o upload para o Dropbox
                        if upload_file_to_dropbox(csv_bytes, full_dropbox_path):
                            st.success(f"🎉 Sucesso! O relatório '{new_filename}' foi processado e salvo no sistema.")
                            st.dataframe(cleaned_df.head()) # Mostra uma prévia dos dados limpos
                            st.balloons()
                        else:
                            st.error("Ocorreu um problema no envio para o Dropbox. Verifique as mensagens de erro.")

    if page == "Visão Geral":
        if not company_for_metrics:
            st.error("Selecione uma empresa para visualizar a Visão Geral.")
        else:
            dre_sel, bal_sel = load_and_clean(company_for_metrics, date_str)
            if dre_sel is None or bal_sel is None:
                st.error("Não há dados para Visão Geral.")
            else:
                rpt = compute_indicators(dre_sel, bal_sel)
                st.header(f"🏁 Indicadores {company_for_metrics} em {date_str}")

                # transforma em Series para lookup por nome
                vals = rpt.set_index("Indicador")["Valor"]
                if "Lucro Bruto" not in vals.index and "Lucro Bruto (approx.)" in vals.index:
                    vals = vals.rename(index={"Lucro Bruto (approx.)": "Lucro Bruto"})

                c1, c2, c3 = st.columns(3)
                c1.metric("Lucro Bruto",   f"R$ {vals['Lucro Bruto']:,.2f}")
                c2.metric("EBITDA",        f"R$ {vals['EBITDA']:,.2f}")
                c3.metric("Lucro Líquido", f"R$ {vals['Lucro Líquido']:,.2f}")

                c4, c5, c6 = st.columns(3)
                c4.metric(
                    "Liquidez Corrente",
                    f"{vals['Liquidez Corrente']:.2f}"
                    if pd.notnull(vals['Liquidez Corrente']) else "—"
                )
                c5.metric(
                    "Endividamento",
                    f"{vals['Endividamento']:.2%}"
                    if pd.notnull(vals['Endividamento']) else "—"
                )
                c6.metric(
                    "ROE",
                    f"{vals['ROE']:.2%}"
                    if pd.notnull(vals['ROE']) else "—"
                )
                st.markdown("---")

                def format_val(row):
                    val = row["Valor"]
                    ind = row["Indicador"]
                    if ind in ["Lucro Bruto", "EBITDA", "Lucro Líquido"]:
                        return f"R$ {val:,.2f}"
                    elif ind in ["Liquidez Corrente", "Liquidez Seca"]:
                        return f"{val:,.2f}" if pd.notnull(val) else "-"
                    else:
                        return f"{val:.2%}" if pd.notnull(val) else "-"
                
                rpt_disp = rpt.copy()
                rpt_disp["Valor"] = rpt_disp.apply(format_val, axis=1)

                st.dataframe(rpt_disp, use_container_width=True)

    elif page == "Dashboards":
        if df_all.empty:
            st.info("Nenhum dado disponível para as seleções atuais.")
        else:
            st.header("📈 Dashboards")
            base = alt.Chart(df_all).mark_bar().encode(
                x="account_std:N",
                y="amount:Q",
                color="company_id:N",
                tooltip=["company_id","account_std","amount"]
            )
            st.altair_chart(base, use_container_width=True)

            # 1) Composição do Ativo (pizza)
            st.subheader("Composição do Ativo")
            df_asset = df_all.query("statement=='balance_sheet' and amount>0")
            pie_asset = df_asset.groupby("account_std")["amount"].sum().reset_index()
            fig_asset = px.pie(
                pie_asset, 
                names="account_std", 
                values="amount", 
                title="Ativos por Conta"
            )
            st.plotly_chart(fig_asset, use_container_width=True)

            # 2) Composição do Passivo (pizza)
            st.subheader("Composição do Passivo")
            df_liab = df_all.query("statement=='balance_sheet' and amount<0").copy()
            df_liab["amount"] = df_liab["amount"].abs()
            pie_liab = df_liab.groupby("account_std")["amount"].sum().reset_index()
            fig_liab = px.pie(
                pie_liab, 
                names="account_std", 
                values="amount", 
                title="Passivos por Conta"
            )
            st.plotly_chart(fig_liab, use_container_width=True)

            # 3) Pareto de Despesas Operacionais
            st.subheader("Pareto de Despesas Operacionais")
            df_desp = df_all.query(
                "statement=='income_statement' and amount<0"
            ).copy()
            df_desp["abs_amount"] = df_desp["amount"].abs()
            pareto = (
                df_desp.groupby("account_std")["abs_amount"]
                .sum()
                .reset_index()
                .sort_values("abs_amount", ascending=False)
            )
            pareto["cum_pct"] = pareto["abs_amount"].cumsum() / pareto["abs_amount"].sum()
            bars = alt.Chart(pareto).mark_bar().encode(
                x="account_std:N",
                y="abs_amount:Q"
            )
            line = alt.Chart(pareto).mark_line(color="red").encode(
                x="account_std:N",
                y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%"))
            )
            combo = alt.layer(bars, line).resolve_scale(y="independent")
            st.altair_chart(combo, use_container_width=True)

    else:  # TaxbaseAI
        st.markdown(
            """
            <style>
            .stApp {
              background-color: #E1E3EBFF;
              font-family: 'Segoe UI', sans-serif;
            }
            .chat-history {
              max-height: 60vh;
              overflow-y: auto;
              padding-right: 6px;
              margin-bottom: 10px;
            }
            .typing-indicator {
              font-style: italic;
              color: #888;
            }
            .stChatMessage.user {
              background-color: #d1e7ff;
              border-radius: 16px;
              padding: 10px 14px;
              color: #003366;
              margin-bottom: 8px;
              max-width: 80%
            }
            .stChatMessage.assistant {
              background-color: #ffffff;
              border-radius: 16px;
              padding: 10px 14px;
              border: 1px solid #e0e0e0;
              color: #222;
              margin-bottom: 8px;
              max-width: 80%;
            }
            .stChatMessage.user { margin-left: auto; }
            .stChatMessage.assistant { margin-right: auto; }
            .suggestion-btn {
              display: inline-block;
              margin: 4px 6px 0 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        def enqueue_prompt(q: str):
            st.session_state["queued_prompt"] = q
            st.session_state.pop("suggestions", None)
        
        st.header(f"🤖 TaxbaseAI | Sua AI Contábil - {company_for_metrics if company_for_metrics else ''}")

        if "messages" not in st.session_state:
            st.session_state.messages = []
            greeting = initial_greeting(company_for_metrics, date_str) if company_for_metrics else "Oi — selecione uma empresa para começarmos."
            st.session_state.messages.append({
                "role": "assistant",
                "content": greeting,
                "avatar": "🤖"
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("Quer começar por aqui?")
                cols = st.columns(3)
                for i, q in enumerate([
                    "Me traga um raio-x financeiro do período",
                    "Como está a liquidez e a alavancagem?",
                    "Quais despesas mais subiram e por quê?"
                ]):
                    cols[i].button(q, key=f"starter_{i}", on_click=enqueue_prompt, args=(q,))

        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(
                msg["role"],
                avatar=msg.get("avatar", "🤖" if msg["role"] == "assistant" else "🧑")
            ):
                st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        queued = st.session_state.pop("queued_prompt", None)
        prompt = queued if queued else st.chat_input("Digite sua pergunta sobre os indicadores...")

        if prompt:
            st.session_state.pop("suggestions", None)
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑"})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            
            contexts = semantic_search(prompt, index, meta, top_k=3)
            ctx_txt = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in contexts)

            brief_ctx = brief_history(st.session_state.messages)

            dre_raw = load_monthly_csv_from_dropbox(
                prefix_month=f"DRE_{date_str}",
                company_id=company_for_metrics,
                expected_cols=["nome_empresa", "descrição", "valor"]
            )
            bal_raw = load_monthly_csv_from_dropbox(
                prefix_month=f"BALANCO_{date_str}",
                company_id=company_for_metrics,
                expected_cols=["nome_empresa", "descrição", "saldo_atual"]
            )
            if dre_raw is None or bal_raw is None:
                st.error("Não foi possível carregar os dados brutos.")
                st.stop()

            dre_csv = dre_raw.to_csv(index=False)
            bal_csv = bal_raw.to_csv(index=False)

            full_prompt = f"""
Você é um assistente contábil.

Sistema (tom e contexto resumido):
{TONE_SYSTEM}

Histórico Resumido:
{brief_ctx}

Aqui estão os dados brutos da Demonstração de Resultados (DRE):
{dre_csv}

E aqui os dados brutos do Balanço Patrimonial:
{bal_csv}

Contextos anteriores (semânticos):
{ctx_txt}

Pergunta: {prompt}

Responda de forma objetiva e fundamentada nos dados brutos acima.
"""

            with st.chat_message("assistant", avatar="🤖"):
                typing_placeholder = st.empty()
                for i in range(10):
                    dots = "." * ((i % 3) + 1)
                    typing_placeholder.markdown(f"<span class='typing-indicator'>sendo digitado{dots}</span>", unsafe_allow_html=True)
                    time.sleep(0.15)

                resposta = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Assistente contábil de indicadores."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0
                ).choices[0].message.content.strip()

                typing_placeholder.empty()
                stream_placeholder = st.empty()
                displayed_text = ""
                for ch in resposta:
                    displayed_text += ch
                    stream_placeholder.markdown(displayed_text)
                    time.sleep(0.003)

            st.session_state.messages.append({"role": "assistant", "content": resposta, "avatar": "🤖"})

            suggestions = generate_followups(prompt, resposta, company_for_metrics, date_str)

            st.session_state["suggestions"] = suggestions or []

            upsert_embedding(prompt, resposta, index, meta)
        
        if st.session_state.get("suggestions"):
            st.markdown("**Sugestões para continuar:**")
            cols = st.columns(min(3, len(st.session_state["suggestions"])))
            for i, q in enumerate(st.session_state["suggestions"]):
                cols[i % len(cols)].button(q, key=f"followup_{i}", on_click=enqueue_prompt, args=(q,))

elif authentication_status is False:
    st.error("Usuário ou senha incorretos")
else:
    st.info("Por favor, faça login para continuar")