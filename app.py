import streamlit as st
import calendar
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
from sqlalchemy import text
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import git
import os
import io
import xlrd
import re

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
        df_users = pd.read_sql("SELECT username, name, password, role FROM usuarios", engine)
        df_access = pd.read_sql("SELECT username, company_name, is_default FROM acesso_empresas", engine)
    except Exception as e:
        st.error(f"Erro ao ler usuários ou acessos do banco: {e}")
        return {}
        
    users = {}
    for _, row in df_users.iterrows():
        user_access_df = df_access[df_access["username"] == row["username"]]
        user_companies = user_access_df["company_name"].tolist()
        
        # Encontra a empresa padrão (onde is_default == 1)
        default_company_series = user_access_df[user_access_df["is_default"] == 1]["company_name"]
        default_company = default_company_series.iloc[0] if not default_company_series.empty else None
        
        users[row["username"]] = {
            "name": row["name"], "password": row["password"], "role": row["role"],
            "empresas": user_companies,
            "default_company": default_company # Nova chave com a empresa padrão
        }
    return users

def add_user_to_db(username: str, name: str, password: str, empresas: list[str], role: str, default_company: str | None):
    """Adiciona um novo usuário ao banco de dados, falhando se o usuário já existir."""
    try:
        # Verifica se o usuário já existe
        existing_users_df = pd.read_sql("SELECT username FROM usuarios WHERE username = ?", engine, params=(username,))
        if not existing_users_df.empty:
            st.error(f"Erro: O usuário '{username}' já existe. Use a área de edição para modificá-lo.")
            return False
            
        # Garante que uma senha foi digitada para novos usuários
        if not password:
            st.error("É necessário fornecer uma senha para criar um novo usuário.")
            return False
        
        # Cria o novo usuário
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        new_user = pd.DataFrame([{"username": username, "name": name, "password": hashed_password, "role": role}])
        new_user.to_sql("usuarios", engine, if_exists="append", index=False)

        # Adiciona as permissões na tabela de acesso
        if empresas:
            access_data = [{"username": username, "company_name": emp, "is_default": 1 if emp == default_company else 0} for emp in empresas]
            df_access = pd.DataFrame(access_data)
            df_access.to_sql("acesso_empresas", engine, if_exists="append", index=False)
        
        return True

    except Exception as e:
        st.error(f"Ocorreu um erro ao criar o usuário: {e}")
        return False
    
def update_user_in_db(username: str, new_name: str, new_empresas: list[str], new_role: str, default_company: str | None):
    """Atualiza os dados de um usuário existente e suas permissões de empresa."""
    try:
        with engine.connect() as connection:
            # 1. Atualiza o nome e o perfil na tabela 'usuarios'
            stmt_user = text("UPDATE usuarios SET name = :name, role = :role WHERE username = :username")
            connection.execute(stmt_user, {"name": new_name, "role": new_role, "username": username})

            # 2. Apaga TODAS as permissões de empresa antigas deste usuário
            stmt_delete = text("DELETE FROM acesso_empresas WHERE username = :username")
            connection.execute(stmt_delete, {"username": username})

            # 3. Insere as NOVAS permissões de empresa
            if new_empresas:
                # Prepara os dados para inserção em lote
                access_data = [{"username": username, "company_name": emp, "is_default": 1 if emp == default_company else 0} for emp in new_empresas]
                stmt_insert = text("INSERT INTO acesso_empresas (username, company_name, is_default) VALUES (:username, :company_name, :is_default)")
                connection.execute(stmt_insert, access_data)
            
            # Confirma todas as transações
            connection.commit()
        return True
    except Exception as e:
        st.error(f"Ocorreu um erro ao atualizar o usuário: {e}")
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
    """Carrega todos os usuários para exibição (sem a antiga coluna 'empresa')."""
    try:
        # CORREÇÃO: Seleciona apenas as colunas que ainda existem
        return pd.read_sql("SELECT username, name, role FROM usuarios", engine)
    except Exception as e:
        # Melhoria: Mostra o erro se algo der errado no futuro
        st.error(f"Erro ao carregar a lista de usuários: {e}")
        return pd.DataFrame()

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
                <li><strong>Senha para Acesso:</strong> {temp_password}</li>
            </ul>
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

def process_contas_a_pagar_csv(uploaded_file, company_name: str) -> pd.DataFrame | None:
    """Processa arquivos de Contas a Pagar de diferentes formatos."""
    try:
        # Lógica de leitura (Excel ou CSV)
        df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith(('.xls', '.xlsx')) else pd.read_csv(uploaded_file)
        
        rename_map = None
        
        # --- Lógica "Camaleão" para identificar o formato do arquivo ---
        
        # Formato 1 (o primeiro que você enviou)
        if all(col in df.columns for col in ['DATA DE VENCIMENTO', 'SALDO A PAGAR', 'NOME DO FORNECEDOR']):
            rename_map = {
                'NOME DO FORNECEDOR': 'fornecedor',
                'DATA DE VENCIMENTO': 'vencimento',
                'SALDO A PAGAR': 'saldo',
                'EMPRESA': 'company'
            }
        
        # Formato 2 (o segundo que você enviou)
        elif all(col in df.columns for col in ['Dt. Contabil', 'Valor', 'Razão Social']):
            rename_map = {
                'Razão Social': 'fornecedor',
                'Dt. Contabil': 'vencimento',
                'Valor': 'saldo',
                'Fantasia': 'company'
            }

        # --- NOVO BLOCO ADICIONADO AQUI ---
        # Formato 3 (o mais recente)
        elif all(col in df.columns for col in ['Fornecedor', 'Pagamento', 'Valor pago']):
            rename_map = {
                'Fornecedor': 'fornecedor',
                'Pagamento': 'vencimento', # Usando data de pagamento para a análise de histórico
                'Valor pago': 'saldo'
                # A coluna 'company' não parece estar neste arquivo, será adicionada depois
            }
        # --- FIM DO NOVO BLOCO ---

        if rename_map is None:
            st.error(f"Arquivo de Contas a Pagar inválido. Não foi possível identificar as colunas necessárias. Colunas encontradas: {df.columns.tolist()}")
            return None
            
        df_clean = df.rename(columns=rename_map)

        # Converte as colunas para os tipos corretos
        df_clean['vencimento'] = pd.to_datetime(df_clean['vencimento'], errors='coerce')
        df_clean['saldo'] = pd.to_numeric(df_clean['saldo'], errors='coerce')
        
        # Garante que a coluna 'company' exista, se não foi mapeada
        if 'company' not in df_clean.columns:
            df_clean['company'] = company_name

        df_clean = df_clean.dropna(subset=['vencimento', 'saldo', 'fornecedor'])
        
        return df_clean[['company', 'fornecedor', 'vencimento', 'saldo']]

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo de Contas a Pagar. Detalhe: {e}")
        return None

# (Substitua a sua função load_data_for_period por esta)
def load_data_for_period(companies: list, start_date, end_date) -> dict:
    """
    Carrega e consolida os dados de DRE, Balanço e Contas a Pagar para um período,
    retornando um dicionário com os DataFrames separados.
    """
    all_dre, all_bal, all_ap = [], [], []
    
    month_range = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m").tolist()

    for comp in companies:
        for month_str in month_range:
            # Carrega DRE e Balanço
            dre, bal = load_and_clean(comp, month_str)
            if dre is not None: all_dre.append(dre)
            if bal is not None: all_bal.append(bal)

            # Carrega Contas a Pagar
            df_ap_month = load_monthly_csv_from_dropbox(
                prefix_month=f"CONTASAPAGAR_{month_str}",
                company_id=comp,
                expected_cols=['company', 'fornecedor', 'vencimento', 'saldo']
            )
            if df_ap_month is not None:
                all_ap.append(df_ap_month)

    # Consolida cada tipo de dado em seu próprio DataFrame
    return {
        "dre": pd.concat(all_dre, ignore_index=True) if all_dre else pd.DataFrame(),
        "bal": pd.concat(all_bal, ignore_index=True) if all_bal else pd.DataFrame(),
        "ap": pd.concat(all_ap, ignore_index=True) if all_ap else pd.DataFrame()
    }

def process_accounting_csv(uploaded_file, company_name: str) -> pd.DataFrame | None:
    """
    Processa arquivos contábeis (DRE ou Balancete) de diferentes formatos, 
    adaptando-se às colunas encontradas.
    """
    try:
        file_name = uploaded_file.name
        df = None

        # Lógica de leitura do arquivo (permanece a mesma)
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

        # --- LÓGICA "CAMALEÃO" ATUALIZADA ---
        rename_map = None

        # Procura pelo formato com 'saldoatu_cta'
        if "nome_cta" in df.columns and "saldoatu_cta" in df.columns:
            rename_map = {"nome_cta": "account", "saldoatu_cta": "amount"}
        
        # ADICIONADO: Procura pelo formato com 'saldoatu' (sem o _cta)
        elif "nome_cta" in df.columns and "saldoatu" in df.columns:
            rename_map = {"nome_cta": "account", "saldoatu": "amount"}

        # Procura pelo formato com 'nomeconta'
        elif "nomeconta" in df.columns and "valor" in df.columns:
            rename_map = {"nomeconta": "account", "valor": "amount"}
        
        # Procura pelo formato com 'Descrição' (para arquivos mais antigos)
        elif "Descrição" in df.columns and "Valor" in df.columns:
             rename_map = {"Descrição": "account", "Valor": "amount"}

        if rename_map is None:
            st.error(f"Não foi possível identificar as colunas de 'conta' e 'valor' neste arquivo. Colunas encontradas: {df.columns.tolist()}")
            return None
        # ----------------------------------------------------

        # O resto do código continua o mesmo...
        df = df.rename(columns=rename_map)
        df = df.dropna(subset=['amount'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        df['company'] = company_name
        
        return df[["company", "account", "amount"]]

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Detalhe: {e}")
        return None

def upload_file_to_dropbox(file_bytes: bytes, dropbox_path: str) -> bool:
    """
    Faz o upload de um conteúdo em bytes para um caminho específico no Dropbox, sobrescrevendo se o arquivo já existir.
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
def load_csv_from_dropbox(filename: str, expected_cols: list[str] | None) -> pd.DataFrame | None:
    path = f"{BASE_PATH}/{filename}"
    try:
        _, res = dbx.files_download(path=path)
    except dropbox.exceptions.ApiError:
        st.warning(f"Arquivo não encontrado: {filename}")
        return None
    df = pd.read_csv(BytesIO(res.content))
    
    # --- CORREÇÃO ADICIONADA AQUI ---
    # Só faz a verificação de colunas se 'expected_cols' não for None
    if expected_cols is not None:
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
def load_monthly_csv_from_dropbox(prefix_month: str, company_id: str, expected_cols: list[str] | None) -> pd.DataFrame | None:
    """
    Carrega o arquivo mensal mais recente de uma empresa, usando regex para uma busca robusta.
    """
    try:
        entries = dbx.files_list_folder(BASE_PATH).entries
        all_filenames = [e.name for e in entries]
    except dropbox.exceptions.ApiError:
        st.warning(f"Erro ao listar arquivos do Dropbox em {BASE_PATH}")
        return None

    # --- LÓGICA DE BUSCA ATUALIZADA COM REGEX ---
    # Constrói um padrão de regex:
    # Procura por um arquivo que COMEÇA com o prefixo do mês,
    # tem QUALQUER COISA no meio, e TERMINA com o sufixo da empresa.
    # re.IGNORECASE torna a busca insensível a maiúsculas/minúsculas.
    pattern = re.compile(f"^{re.escape(prefix_month)}.*_{re.escape(company_id)}\\.csv$", re.IGNORECASE)
    
    candidates = [name for name in all_filenames if pattern.search(name)]
    # --- FIM DA ATUALIZAÇÃO ---

    if not candidates:
        st.warning(f"Nenhum arquivo encontrado com o padrão: '{prefix_month}..._{company_id}.csv'")
        return None
    
    # Se houver múltiplos candidatos, pega o último em ordem alfabética
    latest = sorted(candidates)[-1]
    return load_csv_from_dropbox(latest, expected_cols)

def load_and_clean(company_id: str, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Carrega os arquivos DRE e BALANCO já processados do Dropbox.
    Esta função agora espera o formato limpo (company, account, amount)
    que é salvo pelo painel de administrador.
    """
    # As colunas esperadas agora são as colunas padronizadas
    expected_cols = ["company", "account", "amount"]

    # Carrega o DRE
    dre_df = load_monthly_csv_from_dropbox(
        prefix_month=f"DRE_{date_str}",
        company_id=company_id,
        expected_cols=expected_cols
    )
    
    # Carrega o BALANCO
    bal_df = load_monthly_csv_from_dropbox(
        prefix_month=f"BALANCO_{date_str}",
        company_id=company_id,
        expected_cols=expected_cols
    )

    if dre_df is None or bal_df is None:
        return None, None

    # Como os dados já estão limpos, só precisamos adicionar os metadados
    dre = add_metadata(dre_df, "income_statement", date_str, company_id)
    bal = add_metadata(bal_df, "balance_sheet", date_str, company_id)

    # Renomeia a coluna 'account' para 'account_std' para manter compatibilidade
    # com o resto do código que espera essa coluna.
    dre["account_std"] = dre["account"]
    bal["account_std"] = bal["account"]

    # A lógica de tratar despesas operacionais ainda é necessária
    dre.loc[dre["account_std"] == "operating_expenses", "amount"] = dre.loc[dre["account_std"] == "operating_expenses", "amount"].abs()

    return dre, bal

def check_file_exists_for_month(company_id: str, month_str: str) -> bool:
    """Verifica se um arquivo DRE ou BALANCO existe para uma empresa e mês específicos."""
    try:
        entries = dbx.files_list_folder(BASE_PATH).entries
        filenames = [e.name for e in entries]
        
        # Procura por DRE ou Balanço do mês e empresa especificados
        dre_exists = f"DRE_{month_str}_{company_id}.csv" in filenames
        balanco_exists = f"BALANCO_{month_str}_{company_id}.csv" in filenames
        
        return dre_exists or balanco_exists # Retorna True se qualquer um dos dois existir
        
    except dropbox.exceptions.ApiError:
        return False

def find_latest_available_month(company_id: str) -> str | None:
    """
    Verifica o Dropbox e retorna o último mês (formato 'YYYY-MM')
    que possui um arquivo DRE ou BALANCO para a empresa especificada.
    """
    try:
        entries = dbx.files_list_folder(BASE_PATH).entries
    except dropbox.exceptions.ApiError:
        return None

    # --- LINHA ALTERADA PARA FILTRAR APENAS DRE E BALANCO ---
    suffix = f"_{company_id}.csv"
    user_files = [
        e.name for e in entries 
        if e.name.endswith(suffix) and (e.name.startswith("DRE_") or e.name.startswith("BALANCO_"))
    ]
    
    # Usa expressão regular para encontrar o padrão de data 'AAAA-MM' no nome do arquivo
    date_pattern = re.compile(r"_(\d{4}-\d{2})_")
    dates = []
    
    for filename in user_files:
        match = date_pattern.search(filename)
        if match:
            dates.append(match.group(1))
            
    if not dates:
        return None

    return max(dates)
# -----------------------------------------------------------------------------
# 3. Cálculo de Indicadores
# -----------------------------------------------------------------------------
def compute_indicators(dre: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    # --- DRE: Lógica de cálculo com EBIT e EBITDA separados ---
    d = dre.copy()
    d["DESC"] = d["account"].astype(str).str.strip().str.upper()

    def sum_dre(regex: str, absolute: bool = False) -> float:
        vals = d.loc[d["DESC"].str.contains(regex, regex=True, na=False), "amount"]
        return vals.abs().sum() if absolute else vals.sum()

    # 1) Receita Líquida
    receitas_brutas = sum_dre(r"RECEITA BRUTAS|RECEITA DE PRESTAÇÃO")
    deducoes_impostos = sum_dre(r"CANCELAMENTO E DEVOLUÇÕES|IMPOSTOS SOBRE VENDAS", absolute=True)
    receita_liq = receitas_brutas - deducoes_impostos

    # 2) Custos
    custos = sum_dre(r"MATERIAL APLICADO|SERVICOS TOMADOS", absolute=True)

    # 3) Lucro Bruto
    lucro_bruto = receita_liq - custos

    # 4) Despesas Operacionais
    despesas_op = sum_dre(r"DESPESAS COM ENTREGA|DESPESAS GERAIS", absolute=True)
    
    # 5) Lucro Operacional (EBIT)
    lucro_operacional_ebit = lucro_bruto - despesas_op

    # 6) Depreciação (será 0 se não encontrar a conta, o que é o caso atual)
    deprec = sum_dre(r"DEPRECIA", absolute=True)

    # 7) EBITDA (EBIT + Depreciação)
    ebitda = lucro_operacional_ebit + deprec

    # 8) Lucro Líquido (Aproximação baseada no EBIT, pois não há juros/impostos)
    lucro_liq = lucro_operacional_ebit

    # --- Balanço Patrimonial (Lógica mantida) ---
    b = bal.copy()
    b["DESC"] = b["account"].astype(str).str.strip().str.upper()

    def sum_bal(regex: str) -> float:
        return b.loc[b["DESC"].str.contains(regex, regex=True, na=False), "amount"].abs().sum()

    ativo_circ = sum_bal(r"^ATIVO CIRCULANTE$")
    pass_circ  = sum_bal(r"^PASSIVO CIRCULANTE$")
    estoque    = sum_bal(r"^ESTOQUES$")
    liquidez_corrente = ativo_circ / pass_circ if pass_circ else 0
    liquidez_seca     = (ativo_circ - estoque) / pass_circ if pass_circ else 0
    total_ativo = sum_bal(r"^ATIVO$")
    pass_ncirc = sum_bal(r"PASSIVO N[AÃ]O CIRCULANTE")
    total_pass = pass_circ + pass_ncirc
    endividamento = total_pass / total_ativo if total_ativo else 0
    patr_liq = sum_bal(r"^PATRIMONIO LIQUIDO$")
    if patr_liq == 0:
        patr_liq = total_ativo - total_pass
    roa = lucro_liq / total_ativo if total_ativo else 0
    roe = lucro_liq / patr_liq if patr_liq else 0

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

    with st.sidebar:
        st.image("assets/taxbaseAI_logo.png", width=250)
        st.divider()

    authenticator.logout("Sair", "sidebar")
    st.sidebar.success(f"Conectado como {user_info['name']} ({role})")

    user_info = USERS[username]
    role = user_info["role"]

    if role == 'admin':
        # Se for admin, ele pode ver TODAS as empresas cadastradas no sistema
        accessible_companies = load_companies_from_db()
    else:
        # Se for um usuário normal, ele pode ver apenas as empresas associadas a ele
        accessible_companies = user_info["empresas"]

    if not accessible_companies:
        st.sidebar.error("Seu usuário não tem acesso a nenhuma empresa.")
        st.stop()
    
    # --- LÓGICA DE SELEÇÃO SIMPLIFICADA ---
    # Encontra a empresa padrão do usuário, definida no cadastro
    user_default_company = user_info.get("default_company")
    default_index = 0
    
    # Tenta encontrar o índice da empresa padrão na lista de empresas que o usuário pode acessar
    if user_default_company and user_default_company in accessible_companies:
        default_index = accessible_companies.index(user_default_company)

    # Cria o seletor único de empresa, já pré-selecionado com a padrão
    company_for_metrics = st.sidebar.selectbox(
        "Empresa", # Label simplificado para clareza
        accessible_companies,
        index=default_index
    )

    # A variável 'session_companies' agora conterá apenas a empresa única selecionada
    # Isso garante que os dashboards também foquem nesta empresa
    session_companies = [company_for_metrics] if company_for_metrics else []

    ## --- Lógica Inteligente para o Período de Análise ---
    if company_for_metrics:
        default_company_for_period = company_for_metrics
    elif accessible_companies:
        default_company_for_period = accessible_companies[0]
    else:
        st.sidebar.warning("Nenhuma empresa disponível para definir o período.")
        st.stop()
    
    today = datetime.now().date()

    # 1. REGRA DO DIA 10: Define o mês alvo
    if today.day < 10:
        target_date = today - relativedelta(months=1)
    else:
        target_date = today

    # 2. REGRA DO ÚLTIMO ARQUIVO: Verifica se o alvo tem dados, se não, busca o último
    target_month_str = target_date.strftime("%Y-%m")

    if check_file_exists_for_month(default_company_for_period, target_month_str):
        default_end_date = target_date
    else:
        latest_month_str = find_latest_available_month(default_company_for_period)
        if latest_month_str:
            default_end_date = datetime.strptime(latest_month_str, "%Y-%m").date()
        else:
             default_end_date = target_date
    
    # O período inicial padrão é 5 meses antes do final
    default_start_date = default_end_date

    # --- Interface do Seletor (Usa os valores padrão que calculamos) ---
    st.sidebar.markdown("##### Período de Análise")
    year_list = list(range(today.year + 1, today.year - 6, -1))
    month_list = list(range(1, 13))

    # --- Cria a interface com 4 seletores em colunas ---
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_month = st.selectbox("Mês Inicial", month_list, index=month_list.index(default_start_date.month), format_func=lambda m: f"{m:02d}")
        end_month = st.selectbox("Mês Final", month_list, index=month_list.index(default_end_date.month), format_func=lambda m: f"{m:02d}")
    with col2:
        start_year = st.selectbox("Ano Inicial", year_list, index=year_list.index(default_start_date.year))
        end_year = st.selectbox("Ano Final", year_list, index=year_list.index(default_end_date.year))

    # --- Monta as datas de início e fim com base na seleção ---
    try:
        start_period = datetime(start_year, start_month, 1).date()
        _, last_day = calendar.monthrange(end_year, end_month)
        end_period = datetime(end_year, end_month, last_day).date()

        if start_period > end_period:
            st.sidebar.error("A data inicial não pode ser posterior à data final.")
            st.stop()

    except Exception as e:
        st.sidebar.error("Período inválido. Verifique as datas.")
        st.stop()

    date_str = end_period.strftime("%Y-%m")

    # Carregar dados com tratamento de erros
    all_dre, all_bal = [], []
    for comp in session_companies:
        dre, bal = load_and_clean(comp, date_str)
        if dre is None or bal is None:
            st.warning(f"Pulando {comp}: dados não encontrados.")
            continue
        all_dre.append(dre)
        all_bal.append(bal)

    if session_companies:
        df_all = load_data_for_period(session_companies, start_period, end_period)
    else:
        df_all = pd.DataFrame()

    page = st.sidebar.radio("📊 Navegação", ["Dashboards", "TaxbaseAI"]) # Após ajustes adicionar o Visão Geral de volta

    if role == "admin":
        if st.sidebar.button("Painel do Administrador"):
            st.session_state.page = "Admin"

        if st.session_state.get("page") == "Admin":
            if st.sidebar.button("⬅️ Voltar"):
                del st.session_state.page
                st.rerun()

    active_page = st.session_state.get("page", page)

    if active_page == "Admin" and role == "admin":
        st.header("🔑 Painel do Administrador")
        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["Gerenciar Usuários", "Gerenciar Empresas", "📤 Upload de Relatórios"])

        with admin_tab1:
            # --- ÁREA 1: CRIAR NOVO USUÁRIO ---
            st.subheader("➕ Criar Novo Usuário")
            with st.form("new_user_form", clear_on_submit=True):
                st.write("Preencha os dados para cadastrar um novo usuário no sistema.")
                new_username = st.text_input("Usuário (E-mail para login)")
                new_name = st.text_input("Nome Completo")
                new_password = st.text_input("Senha Provisória", type="password")
        
                all_companies = load_companies_from_db()
                assigned_companies = st.multiselect(
                    "Selecione as Empresas de Acesso", 
                    options=all_companies,
                    placeholder="Selecione uma ou mais empresas"
                )

                default_company_create = st.selectbox(
                    "Selecione a Empresa Padrão",
                    options=assigned_companies, # As opções são as empresas já selecionadas
                    index=0 if assigned_companies else None,
                    help="Esta será a empresa pré-selecionada quando o usuário fizer login."
                )

                assigned_role = st.selectbox("Perfil de Acesso (Role)", options=["user", "admin"])
        
                submitted_create = st.form_submit_button("Criar Usuário")
                if submitted_create:
                    if new_username and new_name:
                        if add_user_to_db(new_username, new_name, new_password, assigned_companies, assigned_role, default_company_create):
                            st.success(f"Usuário '{new_name}' criado com sucesso!")
                            git_auto_commit(commit_message=f"feat(db): Adiciona novo usuário '{new_name}'")
                            # A chamada para enviar e-mail continua funcionando aqui
                            send_invitation_email_sendgrid(new_username, new_password)
                    else:
                        st.warning("Usuário e Nome Completo são campos obrigatórios.")

            st.divider()

            # --- ÁREA 2: EDITAR USUÁRIO EXISTENTE ---
            st.subheader("✏️ Editar Usuário Existente")
    
            # Carrega todos os usuários para o seletor
            users_df = get_all_users()
            if not users_df.empty:
                user_to_edit = st.selectbox(
                    "Selecione um usuário para editar", 
                    options=users_df['username'], 
                    index=None,
                    placeholder="Escolha um usuário..."
                )

                if user_to_edit:
                    # Carrega os dados completos do usuário selecionado
                    user_data = USERS.get(user_to_edit)
            
                    with st.form("edit_user_form"):
                        st.write(f"Editando dados de **{user_data.get('name')}** (`{user_to_edit}`)")
                
                        edit_name = st.text_input("Nome Completo", value=user_data.get('name'))
                        edit_role = st.selectbox("Perfil de Acesso (Role)", options=["user", "admin"], index=["user", "admin"].index(user_data.get('role', 'user')))
                
                        edit_companies = st.multiselect(
                            "Empresas de Acesso", 
                            options=all_companies,
                            default=user_data.get('empresas', [])
                        )

                        default_company_edit = st.selectbox(
                            "Selecione a Empresa Padrão",
                            options=edit_companies, # As opções são as empresas selecionadas para edição
                            index=edit_companies.index(user_data.get('default_company')) if user_data.get('default_company') in edit_companies else 0 if edit_companies else None,
                            )
                
                        st.warning("Para redefinir a senha, use a funcionalidade específica (a ser criada).")

                        submitted_edit = st.form_submit_button("Atualizar Usuário")
                        if submitted_edit:
                            if update_user_in_db(user_to_edit, edit_name, edit_companies, edit_role, default_company_edit):
                                st.success(f"Usuário '{edit_name}' atualizado com sucesso!")
                                git_auto_commit(commit_message=f"chore(db): Atualiza dados do usuário '{edit_name}'")
                                st.info("Recarregando em 3 segundos para refletir as mudanças...")
                                time.sleep(3)
                                st.rerun()
                    # A função update_user_in_db já mostra a mensagem de erro internamente
            else:
                st.info("Nenhum usuário cadastrado para editar.")

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
                    options=["DRE", "BALANCO", "CONTASAPAGAR"],
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
                    # --- INÍCIO DO BLOCO DE LÓGICA ATUALIZADO ---
                    cleaned_df = None # Inicia a variável como nula

                    # Decide qual função de processamento usar com base no tipo de relatório
                    if report_type in ["DRE", "BALANCO"]:
                        cleaned_df = process_accounting_csv(uploaded_file, company_to_upload)
                    elif report_type == "CONTASAPAGAR":
                        cleaned_df = process_contas_a_pagar_csv(uploaded_file, company_to_upload)

                    # A função retorna um DataFrame se tudo deu certo, ou None se deu erro
                    if cleaned_df is not None:
                        date_str = f"{selected_year}-{selected_month:02d}"
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
        '''
    elif page == "Visão Geral":

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
        '''
    elif page == "Dashboards":
        st.header("📈 Dashboards de Análise de Resultados")

        data_periodo = load_data_for_period(session_companies, start_period, end_period)
        df_dre = data_periodo.get("dre", pd.DataFrame())

        if df_dre.empty:
            st.info("Nenhum dado de DRE disponível para as seleções atuais.")
        else:
            # Filtra apenas os dados de DRE para estes gráficos
            df_dre['month'] = df_dre['ref_date'].dt.to_period('M').astype(str)

            df_dre['DESC'] = df_dre['account'].astype(str).str.strip().str.upper()

            # --- Gráfico 1: Faturamento Mês a Mês ---
            st.subheader("📊 Faturamento (Receita Líquida) Mês a Mês")
        
            # Calcula a Receita Líquida por mês
            df_dre['receitas'] = df_dre.apply(
                lambda row: row['amount'] if 'RECEITA' in row['DESC'] else 0, axis=1
            )
            df_dre['deducoes'] = df_dre.apply(
                lambda row: abs(row['amount']) if 'CANCELAMENTO' in row['DESC'] or 'IMPOSTOS' in row['DESC'] else 0, axis=1
            )
        
            faturamento_mensal = df_dre.groupby('month').apply(
                lambda x: x['receitas'].sum() - x['deducoes'].sum()
            ).reset_index(name='faturamento')

            chart_fat = alt.Chart(faturamento_mensal).mark_bar().encode(
                x=alt.X('month', title='Mês', sort=None),
                y=alt.Y('faturamento', title='Faturamento (R$)'),
                tooltip=['month', alt.Tooltip('faturamento', format=',.2f')]
            ).properties(
                height=400
            )
            st.altair_chart(chart_fat, use_container_width=True)

            # --- Gráfico 2: Valores Gastos Mês a Mês ---
            st.subheader("📊 Valores Gastos (Custos + Despesas) Mês a Mês")

            df_dre['gastos'] = df_dre.apply(
                lambda row: abs(row['amount']) if 'CUSTO' in row['DESC'] or 'DESPESAS' in row['DESC'] or 'MATERIAL' in row['DESC'] or 'SERVICOS' in row['DESC'] else 0, axis=1
            )
            gastos_mensais = df_dre.groupby('month')['gastos'].sum().reset_index()

            chart_gastos = alt.Chart(gastos_mensais).mark_bar(color='firebrick').encode(
                x=alt.X('month', title='Mês', sort=None),
                y=alt.Y('gastos', title='Gastos (R$)'),
                tooltip=['month', alt.Tooltip('gastos', format=',.2f')]
            ).properties(
                height=400
            )
            st.altair_chart(chart_gastos, use_container_width=True)

            # --- Gráfico 3: Proporcionalidade de Despesas e Custos ---
            st.subheader("📊 Proporcionalidade de Despesas e Custos")
            st.markdown(f"Análise para o período de **{start_period.strftime('%m/%Y')}** a **{end_period.strftime('%m/%Y')}**")

            # Agrupa por conta usando o DataFrame completo do período ('df_dre'), sem filtrar por mês
            pizza_data = df_dre.groupby('account')['gastos'].sum().reset_index()
            pizza_data = pizza_data[pizza_data['gastos'] > 0] # Remove contas sem gastos

            if not pizza_data.empty:
                fig_pie = px.pie(
                    pizza_data,
                    names='account',
                    values='gastos',
                    title=f'Composição Consolidada dos Gastos no Período'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(f"Não foram encontrados dados de gastos para o período selecionado.")

    elif page == "TaxbaseAI":
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

            with st.spinner("Analisando dados do período..."):
                # 1. Carrega TODOS os dados do período
                data_periodo = load_data_for_period(session_companies, start_period, end_period)
                dre_raw = data_periodo.get("dre")
                bal_raw = data_periodo.get("bal")
                ap_raw = data_periodo.get("ap")

                if dre_raw is None or dre_raw.empty or bal_raw is None or bal_raw.empty:
                    st.error("Não foi possível carregar os dados essenciais (DRE/Balanço) para a IA.")
                    st.stop()

                # --- INÍCIO DA ATUALIZAÇÃO ---
                # 2. CRIA OS RESUMOS MENSAIS
                # Converte a data para formato de mês para agrupar
                dre_raw['ref_date'] = pd.to_datetime(dre_raw['ref_date']).dt.strftime('%Y-%m')
                bal_raw['ref_date'] = pd.to_datetime(bal_raw['ref_date']).dt.strftime('%Y-%m')

                # Agrupa os dados por mês e conta, somando os valores
                dre_summary = dre_raw.groupby(['ref_date', 'account'])['amount'].sum().reset_index()
                bal_summary = bal_raw.groupby(['ref_date', 'account'])['amount'].sum().reset_index()

                # 3. Converte os RESUMOS para CSV (eles serão muito menores)
                dre_csv = dre_summary.to_csv(index=False)
                bal_csv = bal_summary.to_csv(index=False)
            
                ap_context_str = ""
                if ap_raw is not None and not ap_raw.empty:
                    # Faz o mesmo para o Contas a Pagar
                    ap_raw['vencimento'] = pd.to_datetime(ap_raw['vencimento']).dt.strftime('%Y-%m')
                    ap_summary = ap_raw.groupby(['vencimento', 'fornecedor'])['saldo'].sum().reset_index()
                    ap_csv = ap_summary.to_csv(index=False)
                    ap_context_str = f"\n\nE aqui estão os dados de Contas a Pagar (resumidos por mês e fornecedor):\n{ap_csv}"
            
                # 4. Monta o prompt final, explicando que os dados são resumos
                full_prompt = f"""
                Você é um assistente contábil. Os dados fornecidos foram pré-processados e estão resumidos por mês.

                Período da Análise: de {start_period.strftime('%Y-%m')} a {end_period.strftime('%Y-%m')}.

                Aqui estão os dados da Demonstração de Resultados (DRE), com valores somados por conta e por mês (ref_date):
                {dre_csv}

                E aqui os dados do Balanço Patrimonial, com valores somados por conta e por mês (ref_date):
                {bal_csv}{ap_context_str}
            
                Pergunta do usuário: {prompt}

                Responda de forma objetiva, usando os dados resumidos fornecidos. Ao somar valores, considere todos os meses do período.
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