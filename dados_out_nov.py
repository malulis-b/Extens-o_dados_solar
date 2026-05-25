import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 🛠️ 1. SUAS FUNÇÕES DE CONVERSÃO (BLINDADAS CONTRA VALORES VAZIOS/NAN)
# ==============================================================================

def converter_dia_juliano(coluna_ano, coluna_dia_juliano):
    # Força o ano de estudo como 2001 (ano não-bissexto)
    anos = "2001" 
    # Trata valores vazios/corrompidos convertendo para numérico com segurança
    dias_numericos = pd.to_numeric(coluna_dia_juliano, errors='coerce').fillna(0).astype(int)
    dias = dias_numericos.astype(str)
    datas_reais = pd.to_datetime(anos + "-" + dias, format="%Y-%j", errors='coerce')
    return datas_reais.dt.strftime("%d/%m/%Y")

def converter_hora_minuto(coluna_hora_minuto):
    # Garante o preenchimento de zeros à esquerda (ex: 5 -> 0005 -> 00:05)
    horas_numericas = pd.to_numeric(coluna_hora_minuto, errors='coerce').fillna(0).astype(int)
    horas_str = horas_numericas.astype(str).str.zfill(4)
    return horas_str.str[:2] + ":" + horas_str.str[2:]

# ==============================================================================
# 📂 2. MAPEAMENTO EXCLUSIVO DE ARQUIVOS ÚNICOS (IGNORA CÓPIAS ESPELHADAS)
# ==============================================================================
pasta_2001 = os.path.join("ESOL_FAE", "EstSol", "2001")

arquivos_alvo = {}
for mes in ["out", "OUT", "Out", "nov", "NOV", "Nov"]:
    caminho_busca_dat = os.path.join(pasta_2001, mes, "**", "*.dat")
    caminho_busca_DAT = os.path.join(pasta_2001, mes, "**", "*.DAT")
    
    # Ao usar o nome do arquivo em maiúsculo como chave do dicionário,
    # se o mesmo arquivo (ex: 231001.DAT) existir em duas pastas, ele NÃO é duplicado!
    for f in glob.glob(caminho_busca_dat, recursive=True):
        if os.path.isfile(f): 
            arquivos_alvo[os.path.basename(f).upper()] = f
    for f in glob.glob(caminho_busca_DAT, recursive=True):
        if os.path.isfile(f): 
            arquivos_alvo[os.path.basename(f).upper()] = f

# Transforma de volta em uma lista limpa de caminhos físicos sem repetições
lista_arquivos_unica = list(arquivos_alvo.values())

colunas_curtas = ["DATA", "HORA", "VENTO_VEL", "VENTO_DIR", "TEMP", "RAD_GLOB", "RAD_DIF", "UMIDADE", "PRESSAO", "ARQUIVO"]

colunas_padrao = [
    "ID_Estacao", "Velocidade_Vento_ms", "Direcao_Vento_graus", "Temperatura_C", 
    "Radiacao_Global_W_m2", "Radiacao_Difusa_W_m2", "Rad_Cimel", 
    "Umidade_Relativa_porcentagem", "CR10Temp", "CR10Bat", 
    "Ano", "Dia_Juliano", "Hora_Minuto"
]

lista_tabelas = []
print(f"\n📦 Lendo {len(lista_arquivos_unica)} arquivos .DAT REAIS (sem cópias repetidas)...")

# ==============================================================================
# 📖 3. LEITURA REFORÇADA, CORTE DE CABEÇALHOS E FILTRO TEMPORAL
# ==============================================================================
for arquivo in lista_arquivos_unica:
    nome_arq = os.path.basename(arquivo)
    try:
        # Pula as 4 primeiras linhas de texto descritivo do cabeçalho original
        df_temp = pd.read_csv(arquivo, sep=r'[\s,]+', engine='python', header=None, skiprows=4, on_bad_lines='skip')
        
        if df_temp.empty:
            df_temp = pd.read_csv(arquivo, sep=r'[\s,]+', engine='python', header=None, on_bad_lines='skip')
            
        # Filtro: Mantém apenas linhas onde a primeira coluna seja o ID numérico da estação
        df_temp = df_temp[pd.to_numeric(df_temp[0], errors='coerce').notna()].copy()
        
        if not df_temp.empty:
            num_cols = df_temp.shape[1]
            if num_cols >= len(colunas_padrao):
                df_temp = df_temp.iloc[:, :len(colunas_padrao)]
                df_temp.columns = colunas_padrao
            else:
                df_temp.columns = colunas_padrao[:num_cols]
            
            # 🚨 FILTRO DE CONTROLE DE QUALIDADE: Remove linhas fora de Outubro/Novembro
            # Dias Julianos de Outubro e Novembro (ano não-bissexto) vão estritamente de 274 a 334
            df_temp["Dia_Juliano_Num"] = pd.to_numeric(df_temp["Dia_Juliano"], errors='coerce')
            df_temp = df_temp[(df_temp["Dia_Juliano_Num"] >= 274) & (df_temp["Dia_Juliano_Num"] <= 334)].copy()
            df_temp.drop(columns=["Dia_Juliano_Num"], inplace=True)
            
            if not df_temp.empty:
                df_temp["Arquivo_Origem"] = nome_arq
                lista_tabelas.append(df_temp)
                print(f"   ✓ Lido com sucesso: {nome_arq} ({len(df_temp)} linhas)")
            
    except Exception as e:
        print(f"   ⚠️ Erro crítico ao abrir o arquivo: {nome_arq}")

# ==============================================================================
# 🧼 4. PROCESSAMENTO INTEGRADO, CÁLCULO DE LACUNAS E DADOS DUPLICADOS REAIS
# ==============================================================================
if lista_tabelas:
    df_final = pd.concat(lista_tabelas, ignore_index=True)
    
    # Substituição de erros físicos dos sensores (Ex: 6999, -6999) por valores nulos (NaN)
    df_final.replace([-6999, -6999.0, 6999, 6999.0, -999, -999.0, 999, 999.0, "-6999", "6999", "-999", "999"], np.nan, inplace=True)
    
    # Conversões cronológicas baseadas nas colunas limpas
    df_final["DATA"] = converter_dia_juliano(df_final["Ano"], df_final["Dia_Juliano"])
    df_final["HORA"] = converter_hora_minuto(df_final["Hora_Minuto"])
    
    df_final["_DATA_LINHA_TEMPO"] = pd.to_datetime(df_final["DATA"] + " " + df_final["HORA"], format="%d/%m/%Y %H:%M", errors='coerce')
    df_final.dropna(subset=["_DATA_LINHA_TEMPO"], inplace=True)
    
    # Garante o corte duro para não aceitar nenhum resíduo matemático que escape do escopo de Out/Nov
    df_final = df_final[(df_final["_DATA_LINHA_TEMPO"] >= "2001-10-01") & (df_final["_DATA_LINHA_TEMPO"] <= "2001-11-30 23:59")].copy()
    df_final.sort_values(by="_DATA_LINHA_TEMPO", inplace=True)
    
    # ─── CÁLCULO REAL DE DUPLICADOS (DENTRO DA SÉRIE HISTÓRICA COMBINADA) ───
    total_pontos_brutos = len(df_final)
    linhas_duplicadas = df_final[df_final.duplicated(subset=["DATA", "HORA"], keep=False)].copy()
    total_duplicados_deletar = df_final.duplicated(subset=["DATA", "HORA"], keep="first").sum()
    porcentagem_duplicados = (total_duplicados_deletar / total_pontos_brutos) * 100 if total_pontos_brutos > 0 else 0
    
    if not linhas_duplicadas.empty:
        relatorio_onde_ocorre = linhas_duplicadas.groupby(["DATA", "Arquivo_Origem"]).size().reset_index(name="Pontos_Repetidos")
    else:
        relatorio_onde_ocorre = pd.DataFrame(columns=["DATA", "Arquivo_Origem", "Pontos_Repetidos"])

    # Remove os duplicados reais mantendo o primeiro registro coletado
    df_final.drop_duplicates(subset=["DATA", "HORA"], keep="first", inplace=True)
    total_apos_limpeza = len(df_final)
    
    # Renomeia as colunas internas para o padrão de saída curto solicitado
    df_final.rename(columns={
        "Velocidade_Vento_ms": "VENTO_VEL",
        "Direcao_Vento_graus": "VENTO_DIR",
        "Temperatura_C": "TEMP",
        "Radiacao_Global_W_m2": "RAD_GLOB",
        "Radiacao_Difusa_W_m2": "RAD_DIF",
        "Umidade_Relativa_porcentagem": "UMIDADE",
        "Pressao_Atm": "PRESSAO",
        "Arquivo_Origem": "ARQUIVO"
    }, inplace=True)
    
    df_exibicao = df_final[[c for c in colunas_curtas if c in df_final.columns]].copy()
    df_exibicao["_DATA_LINHA_TEMPO"] = df_final["_DATA_LINHA_TEMPO"]
    
    # Exporta o arquivo de texto consolidado separado por tabulações
    df_exibicao.drop(columns=["_DATA_LINHA_TEMPO"]).to_csv("resultado_out_nov.txt", sep="\t", index=False)
    
    # ─── 📊 CÁLCULO MATEMÁTICO DE LACUNAS E DESCONTINUIDADE REAL ───
    data_inicio = df_exibicao["_DATA_LINHA_TEMPO"].min()
    data_fim = df_exibicao["_DATA_LINHA_TEMPO"].max()
    
    # Gera o índice minuto a minuto ideal esperado para o intervalo encontrado
    idx_perfeito = pd.date_range(start=data_inicio, end=data_fim, freq='1min')
    total_minutos_esperados = len(idx_perfeito)
    
    # Reindexa a série temporal preenchendo os minutos ausentes com NaN (cria as lacunas visuais)
    df_grafico = df_exibicao.copy()
    df_grafico.set_index("_DATA_LINHA_TEMPO", inplace=True)
    df_grafico = df_grafico.reindex(idx_perfeito)
    
    total_lacunas_minutos = df_grafico["DATA"].isna().sum()
    porcentagem_descontinuidade = (total_lacunas_minutos / total_minutos_esperados) * 100 if total_minutos_esperados > 0 else 0
    porcentagem_dados_validos = 100 - porcentagem_descontinuidade

    # ==============================================================================
    # 📊 5. EXIBIÇÃO DO PAINEL METROLÓGICO E GRÁFICO CIENTÍFICO NO TERMINAL
    # ==============================================================================
    print("\n" + "═"*75)
    print(" 📈 EXTRATO DE INTEGRIDADE DA SÉRIE TEMPORAL (OUTUBRO E NOVEMBRO)")
    print("═"*75)
    print(f" ⏱️ Período Real Mapeado: de {data_inicio.strftime('%d/%m/%Y %H:%M')} até {data_fim.strftime('%d/%m/%Y %H:%M')}")
    print(f" 📦 Total de minutos teóricos esperados:       {total_minutos_esperados} minutos.")
    print("═"*75)
    print(f" 🛑 [DESCONTINUIDADE E LACUNAS]")
    print(f"    • Quantidade de minutos em branco (Lacunas):  {total_lacunas_minutos} minutos.")
    print(f"    • PORCENTAGEM DE DESCONTINUIDADE (FALHAS):     {porcentagem_descontinuidade:.3f}%")
    print(f"    • PORCENTAGEM DE DADOS VÁLIDOS (EFETIVIDADE):  {porcentagem_dados_validos:.3f}%")
    print("─"*75)
    print(f" 👥 [DUPLICATAS E REPETIÇÕES INTERNAS]")
    print(f"    • Registros repetidos reais do sensor:        {total_duplicados_deletar} linhas.")
    print(f"    • PORCENTAGEM DE DADOS DUPLICADOS REAL:        {porcentagem_duplicados:.3f}%")
    print(f"    • Total final de minutos limpos e únicos:     {total_apos_limpeza} registros.")
    print("═"*75)
    
    if not relatorio_onde_ocorre.empty and porcentagem_duplicados > 0:
        print("\n🔍 [DETALHE DAS DUPLICADAS POR DIA]")
        print(relatorio_onde_ocorre.to_string(index=False))
        print("═"*75)
        
    print("\n📈 Gerando plotagem gráfica da radiação solar...")
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_grafico.index, df_grafico["RAD_GLOB"], color="orange", label="Radiação Global (W/m²)", alpha=0.8, linewidth=0.8)
    plt.plot(df_grafico.index, df_grafico["RAD_DIF"], color="red", label="Radiação Difusa (W/m²)", alpha=0.6, linewidth=0.8)
    
    plt.title("Evolução do Fluxo de Radiação Solar (Outubro e Novembro de 2001)", fontsize=13, fontweight='bold')
    plt.xlabel("Meses / Dias da Série Temporal", fontsize=11)
    plt.ylabel("Radiação Solar (W/m²)", fontsize=11)
    
    # Trava o limite do eixo X para terminar rigorosamente no fim de Novembro
    plt.xlim(data_inicio, pd.to_datetime("2001-11-30 23:59:00"))
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=4))
    plt.gcf().autofmt_xdate() 
    
    plt.tight_layout()
    plt.savefig("grafico_radiacao_lacunas_corretas.png", dpi=300)
    plt.show()
    
else:
    print("❌ Erro: Nenhum arquivo válido pôde ser processado nas pastas alvo.")