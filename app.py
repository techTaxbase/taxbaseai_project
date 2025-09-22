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
        # Carrega os usuários básicos (sem a coluna 'empresa')
        df_users = pd.read_sql("SELECT username, name, password, role FROM usuarios", engine)
        # Carrega os acessos da nova tabela 'acesso_empresas'
        df_access = pd.read_sql("SELECT username, company_name FROM acesso_empresas", engine)
    except Exception as e:
        st.error(f"Erro ao ler usuários ou acessos do banco: {e}")
        return {}
        
    users = {}
    for _, row in df_users.iterrows():
        # Filtra as empresas para o usuário atual
        user_companies = df_access[df_access["username"] == row["username"]]["company_name"].tolist()
        
        users[row["username"]] = {
            "name": row["name"],
            "password": row["password"],
            "role": row["role"],
            "empresas": user_companies, # A nova chave 'empresas' contém a lista
        }
    return users

def add_user_to_db(username: str, name: str, password: str, empresas: list[str], role: str):
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
            access_data = [{"username": username, "company_name": emp} for emp in empresas]
            df_access = pd.DataFrame(access_data)
            df_access.to_sql("acesso_empresas", engine, if_exists="append", index=False)
        
        return True

    except Exception as e:
        st.error(f"Ocorreu um erro ao criar o usuário: {e}")
        return False
    
def update_user_in_db(username: str, new_name: str, new_empresas: list[str], new_role: str):
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
                access_data = [{"username": username, "company_name": emp} for emp in new_empresas]
                stmt_insert = text("INSERT INTO acesso_empresas (username, company_name) VALUES (:username, :company_name)")
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
        df = pd.read_excel(uploaded_file, engine='openpyxl') if uploaded_file.name.endswith(('.xls', '.xlsx')) else pd.read_csv(uploaded_file)
        
        rename_map = None
        
        # --- Lógica para identificar o formato do arquivo ---
        
        # Formato 1 (o primeiro que você enviou)
        if all(col in df.columns for col in ['DATA DE VENCIMENTO', 'SALDO A PAGAR', 'NOME DO FORNECEDOR']):
            rename_map = {
                'NOME DO FORNECEDOR': 'fornecedor',
                'DATA DE VENCIMENTO': 'vencimento',
                'SALDO A PAGAR': 'saldo',
                'EMPRESA': 'company'
            }
        
        # Formato 2 (o novo arquivo)
        elif all(col in df.columns for col in ['Dt. Contabil', 'Valor', 'Razão Social']):
            st.warning("Aviso: O arquivo não contém 'Data de Vencimento'. Usando 'Dt. Contabil' como substituto. A análise de vencidos pode não estar correta.")
            rename_map = {
                'Razão Social': 'fornecedor',
                'Dt. Contabil': 'vencimento', # Usando como substituto
                'Valor': 'saldo',
                'Fantasia': 'company' # Assumindo que Fantasia pode ser a empresa
            }

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

def load_data_for_period(companies: list, start_date, end_date) -> pd.DataFrame:
    """Carrega e consolida os dados de DRE e Balanço para um período e várias empresas."""
    all_data = []
    
    # Gera uma lista de todos os meses no intervalo
    month_range = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m").tolist()

    for comp in companies:
        for month_str in month_range:
            dre, bal = load_and_clean(comp, month_str)
            if dre is not None:
                all_data.append(dre)
            if bal is not None:
                all_data.append(bal)

    if not all_data:
        return pd.DataFrame() # Retorna um DataFrame vazio se nenhum dado for encontrado
        
    return pd.concat(all_data, ignore_index=True)

def process_accounting_csv(uploaded_file, company_name: str) -> pd.DataFrame | None:
    """
    Processa arquivos contábeis de diferentes formatos, adaptando-se às colunas encontradas.
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

        # --- LÓGICA "CAMALEÃO" PARA ENCONTRAR AS COLUNAS CORRETAS ---
        rename_map = None

        # Procura pelo formato 1 (ex: Balancete)
        if "nome_cta" in df.columns and "saldoatu_cta" in df.columns:
            rename_map = {"nome_cta": "account", "saldoatu_cta": "amount"}
        
        # Procura pelo formato 2 (ex: DRE)
        elif "nomeconta" in df.columns and "valor" in df.columns:
            rename_map = {"nomeconta": "account", "valor": "amount"}
            
        # Adicione outros formatos aqui no futuro, se necessário
        # elif "OutraColunaConta" in df.columns and "OutraColunaValor" in df.columns:
        #     rename_map = {"OutraColunaConta": "account", "OutraColunaValor": "amount"}

        if rename_map is None:
            st.error(f"Não foi possível identificar as colunas de 'conta' e 'valor' neste arquivo. Colunas encontradas: {df.columns.tolist()}")
            return None
        # ----------------------------------------------------------------

        # O resto do código usa o 'rename_map' que foi escolhido acima
        df = df.rename(columns=rename_map)
        
        df = df.dropna(subset=['amount'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
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

    available_companies = load_companies_from_db()
    user_info = USERS[username]
    accessible_companies = user_info["empresas"]

    # O admin vê todas as empresas cadastradas como opções, o usuário normal vê apenas as suas
    options_for_multiselect = load_companies_from_db() if user_info["role"] == "admin" else accessible_companies

    # Garante que as empresas acessíveis sejam válidas
    valid_accessible_companies = [c for c in accessible_companies if c in options_for_multiselect]

    if not options_for_multiselect:
        st.sidebar.warning("Nenhuma empresa disponível para seleção.")
        session_companies = []
    else:
        session_companies = st.sidebar.multiselect(
            "Selecione empresas para a sessão", 
            options=options_for_multiselect, 
            default=valid_accessible_companies
        )

    # Filtrar apenas empresas válidas
    session_companies = [c for c in session_companies if c in available_companies]
    if not session_companies:
        st.sidebar.error("Selecione ao menos uma empresa válida.")

    st.sidebar.markdown("##### Período de Análise")
    # --- Lógica para os valores padrão ---
    today = datetime.now().date()
    default_start = today - relativedelta(months=5)

    # Gera listas de anos e meses para os seletores
    year_list = list(range(today.year + 1, today.year - 6, -1))
    month_list = list(range(1, 13))

    # --- Cria a interface com 4 seletores em colunas ---
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_month = st.selectbox("Mês Inicial", month_list, index=month_list.index(default_start.month), format_func=lambda m: f"{m:02d}")
        end_month = st.selectbox("Mês Final", month_list, index=month_list.index(today.month), format_func=lambda m: f"{m:02d}")
    with col2:
        start_year = st.selectbox("Ano Inicial", year_list, index=year_list.index(default_start.year))
        end_year = st.selectbox("Ano Final", year_list, index=year_list.index(today.year))

    # --- Monta as datas de início e fim com base na seleção ---
    try:
        start_period = datetime(start_year, start_month, 1).date()
        last_day_of_month = calendar.monthrange(end_year, end_month)[1]
        end_period = datetime(end_year, end_month, last_day_of_month).date()

        if start_period > end_period:
            st.sidebar.error("A data inicial não pode ser posterior à data final.")
            st.stop()

    except Exception as e:
        st.sidebar.error("Período inválido. Verifique as datas.")
        st.stop()

    date_str = end_period.strftime("%Y-%m")

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

    if session_companies:
        df_all = load_data_for_period(session_companies, start_period, end_period)
    else:
        df_all = pd.DataFrame()

    page = st.sidebar.radio("📊 Navegação", ["Visão Geral", "Contas a Pagar", "Dashboards", "TaxbaseAI"])

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
                assigned_role = st.selectbox("Perfil de Acesso (Role)", options=["user", "admin"])
        
                submitted_create = st.form_submit_button("Criar Usuário")
                if submitted_create:
                    if new_username and new_name:
                        if add_user_to_db(new_username, new_name, new_password, assigned_companies, assigned_role):
                            st.success(f"Usuário '{new_name}' criado com sucesso!")
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
                
                        st.warning("Para redefinir a senha, use a funcionalidade específica (a ser criada).")

                        submitted_edit = st.form_submit_button("Atualizar Usuário")
                        if submitted_edit:
                            if update_user_in_db(user_to_edit, edit_name, edit_companies, edit_role):
                                st.success(f"Usuário '{edit_name}' atualizado com sucesso!")
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
                        cleaned_df = process_accounting_csv(uploaded_file, company_to_upload, report_type)
                    elif report_type == "CONTASAPAGAR":
                        cleaned_df = process_contas_a_pagar_csv(uploaded_file, company_to_upload)

                    # A função retorna um DataFrame se tudo deu certo, ou None se deu erro
                    if cleaned_df is not None:
                        date_str = end_period.strftime("%Y-%m") 
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

    elif active_page == "Contas a Pagar":
        st.header("💸 Análise de Contas a Pagar")

        # --- Lógica para carregar os dados (permanece a mesma) ---
        all_ap_data = []
        month_range = pd.date_range(start_period, end_period, freq='MS').strftime("%Y-%m").tolist()
        for comp in session_companies:
            for month_str in month_range:
                df_ap_month = load_monthly_csv_from_dropbox(
                    prefix_month=f"CONTASAPAGAR_{month_str}",
                    company_id=comp,
                    expected_cols=None # Lemos sem validar colunas aqui, a validação está no processamento
                )
                if df_ap_month is not None:
                    all_ap_data.append(df_ap_month)

        if not all_ap_data:
            st.info("Nenhum dado de Contas a Pagar encontrado para o período selecionado.")
        else:
            df_ap = pd.concat(all_ap_data, ignore_index=True)
            # Renomeia colunas para o padrão (usando o formato do seu último arquivo)
            # Adicionando uma verificação para evitar erros se as colunas não existirem
            if 'Valor' in df_ap.columns and 'Dt. Contabil' in df_ap.columns:
                df_ap = df_ap.rename(columns={'Valor': 'saldo', 'Dt. Contabil': 'data_contabil', 'Razão Social': 'fornecedor'})
                df_ap['data_contabil'] = pd.to_datetime(df_ap['data_contabil'], errors='coerce')
            else: # Fallback para o primeiro formato de arquivo
                df_ap = df_ap.rename(columns={'vencimento': 'data_contabil'})
                df_ap['data_contabil'] = pd.to_datetime(df_ap['data_contabil'], errors='coerce')

            # --- KPIs ---
            total_pago_periodo = df_ap['saldo'].sum()
            st.metric("Total Contabilizado no Período", f"R$ {total_pago_periodo:,.2f}")

            # --- NOVO GRÁFICO: Histórico de Pagamentos por Mês ---
            st.subheader("🗓️ Histórico de Pagamentos por Mês")
        
            # Cria uma coluna de 'mês' a partir da Data Contábil
            df_ap['mes_contabil'] = df_ap['data_contabil'].dt.to_period('M').astype(str)
            pagamentos_mes = df_ap.groupby('mes_contabil')['saldo'].sum().reset_index()

            chart_pagamentos = alt.Chart(pagamentos_mes).mark_bar().encode(
                x=alt.X('mes_contabil', title='Mês Contábil', sort=None),
                y=alt.Y('saldo', title='Total Pago (R$)'),
                tooltip=['mes_contabil', alt.Tooltip('saldo', format=',.2f')]
            ).properties(height=400)
            st.altair_chart(chart_pagamentos, use_container_width=True)

            # --- Gráfico de Top Fornecedores (Mantido) ---
            st.subheader("🏢 Top 10 Fornecedores")
            if 'fornecedor' in df_ap.columns:
                top_fornecedores = df_ap.groupby('fornecedor')['saldo'].sum().nlargest(10).reset_index()
            
                chart_top_forn = alt.Chart(top_fornecedores).mark_bar().encode(
                    x=alt.X('saldo', title='Saldo Pago (R$)'),
                    y=alt.Y('fornecedor', title='Fornecedor', sort='-x'),
                    tooltip=['fornecedor', alt.Tooltip('saldo', format=',.2f')]
                ).properties(height=500)
                st.altair_chart(chart_top_forn, use_container_width=True)
            else:
                st.warning("Coluna 'fornecedor' não encontrada para gerar o gráfico de Top Fornecedores.")

            # --- Tabela Detalhada (Mantida) ---
            with st.expander("Ver todos os lançamentos do período"):
                st.dataframe(df_ap)

    elif page == "Dashboards":
        st.header("📈 Dashboards de Análise de Resultados")
        if df_all.empty:
            st.info("Nenhum dado de DRE disponível para as seleções atuais.")
        else:
            # Filtra apenas os dados de DRE para estes gráficos
            df_dre = df_all[df_all['statement'] == 'income_statement'].copy()
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
            st.subheader("🍕 Proporcionalidade de Despesas e Custos")
            st.markdown(f"Análise para o último mês do período selecionado: **{end_period.strftime('%m/%Y')}**")

            # Filtra os gastos apenas do último mês
            last_month_str = pd.to_datetime(end_period).to_period('M').strftime('%Y-%m')
            gastos_ultimo_mes = df_dre[df_dre['month'] == last_month_str]
        
            # Agrupa por conta para o gráfico de pizza
            pizza_data = gastos_ultimo_mes.groupby('account')['gastos'].sum().reset_index()
            pizza_data = pizza_data[pizza_data['gastos'] > 0] # Remove contas sem gastos

            if not pizza_data.empty:
                fig_pie = px.pie(
                    pizza_data,
                    names='account',
                    values='gastos',
                    title=f'Composição dos Gastos em {last_month_str}'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(f"Não foram encontrados dados de gastos para o mês {last_month_str}.")

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

            dre_raw = load_monthly_csv_from_dropbox(
                prefix_month=f"DRE_{date_str}",
                company_id=company_for_metrics,
                expected_cols=["company", "account", "amount"]
            )
            bal_raw = load_monthly_csv_from_dropbox(
                prefix_month=f"BALANCO_{date_str}",
                company_id=company_for_metrics,
                expected_cols=["company", "account", "amount"]
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