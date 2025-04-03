import os
import gradio as gr
import time
from datetime import datetime
from Hierarchical_crew import process_optimized_query, query_cache

# Configurações de tema e estilo
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="linear-gradient(to right, #0f2027, #203a43, #2c5364)",
    block_background_fill="#ffffff",
    block_label_background_fill="#f7f7f8",
    block_title_text_color="#ffffff",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_secondary_background_fill="#f3f4f6",
    button_secondary_background_fill_hover="#e5e7eb",
    button_secondary_text_color="#111827",
    block_radius="15px",
    button_small_radius="8px"
)

# Configurações do sistema
SYSTEM_NAME = "CineInteligent"
SYSTEM_AVATAR = "🎬"
USER_AVATAR = "👤"

# Exemplos de perguntas para sugestões rápidas
EXAMPLES = [
    "Quais filmes estão em alta esta semana?",
    "Me recomende filmes de terror psicológico com boa classificação",
    "Quero saber detalhes sobre Star Wars: O Império Contra-Ataca",
    "Quem dirigiu Pulp Fiction e quais outros filmes ele fez?", 
    "Filmes similares a Interestelar"
]

# Função para processar as mensagens - ATUALIZADA para retornar também estatísticas
def process_message(message, history):
    # Verifica se o usuário enviou uma mensagem vazia
    if not message.strip():
        return [{"role": "assistant", "content": "Por favor, digite uma pergunta sobre filmes ou programas de TV."}], get_cache_stats(), get_cache_timestamp()
    
    # Spinner durante o processamento
    yield [{"role": "assistant", "content": "Processando sua consulta... ⌛"}], get_cache_stats(), get_cache_timestamp()
    
    # Processa a consulta usando o sistema hierárquico otimizado
    try:
        response = process_optimized_query(message)
        
        # Verifica se a resposta está em cache
        is_cached = query_cache.get(message) is not None
        cache_indicator = " (resposta em cache)" if is_cached else ""
        
        # Adiciona formatação para links em Markdown
        response = response.replace("https://www.themoviedb.org", "[TMDb](https://www.themoviedb.org")
        response = response.replace(")", ")]")
        
        # Retorna a resposta formatada no formato correto E as estatísticas atualizadas
        yield [{"role": "assistant", "content": response + cache_indicator}], get_cache_stats(), get_cache_timestamp()
    except Exception as e:
        yield [{"role": "assistant", "content": f"Desculpe, ocorreu um erro ao processar sua consulta: {str(e)}\n\nPor favor, tente reformular sua pergunta."}], get_cache_stats(), get_cache_timestamp()

# Função para limpar o histórico e cache - CORRIGIDA e ATUALIZADA
def clear_history_and_cache():
    query_cache.cache = {}
    query_cache.save_cache()
    return [], [{"role": "assistant", "content": "Histórico e cache limpos com sucesso!"}], get_cache_stats(), get_cache_timestamp()

# Função para carregar uma pergunta exemplo
def load_example(example):
    return example

# Função para exibir estatísticas do cache
def get_cache_stats():
    if query_cache.cache:
        num_entries = len(query_cache.cache)
        return f"Cache atual: {num_entries} consultas armazenadas"
    else:
        return "Cache atual: vazio"

# Função para obter timestamp do cache
def get_cache_timestamp():
    if os.path.exists("query_cache.pkl"):
        cache_timestamp = datetime.fromtimestamp(os.path.getmtime("query_cache.pkl"))
        return f"Última atualização: {cache_timestamp.strftime('%d/%m/%Y %H:%M:%S')}"
    return "Cache ainda não criado"

# Interface principal
with gr.Blocks(theme=THEME, title=SYSTEM_NAME) as demo:
    # Cabeçalho
    with gr.Row():
        gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 10px">
            <h1 style="margin-bottom: 10px; color: white; font-size: 2.5rem;">{SYSTEM_NAME} 🎬</h1>
            <h3 style="margin: 5px; color: #e2e8f0;">Seu assistente de cinema e TV com inteligência artificial</h3>
        </div>
        """)
    
    # Chat principal e painel de estatísticas
    with gr.Row():
        # Painel de chat (70% da largura)
        with gr.Column(scale=7):
            # Inicializar chatbot com uma mensagem de boas-vindas
            initial_message = [
                {"role": "assistant", "content": "Olá! Sou o CineInteligent, seu assistente de filmes e programas de TV. Como posso ajudar hoje?"}
            ]
            
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=[SYSTEM_AVATAR, USER_AVATAR],
                height=500,
                type="messages",
                render_markdown=True,
                value=initial_message
            )
            
            # Área de input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Pergunte algo sobre filmes ou programas de TV...",
                    show_label=False,
                    container=False,
                    scale=9
                )
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)
            
            # Exemplos
            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Sugestões de perguntas",
                fn=load_example,
                outputs=msg
            )
        
        # Painel lateral (30% da largura)
        with gr.Column(scale=3):
            with gr.Group():
                gr.HTML("<h3 style='text-align:center; margin-bottom:10px'>Estatísticas & Controles</h3>")
                
                # Estatísticas de cache com atualização automática
                cache_stats = gr.Markdown(get_cache_stats())
                cache_time = gr.Markdown(get_cache_timestamp())
                
                # Botão de limpar cache
                clear_btn = gr.Button("🧹 Limpar Histórico e Cache", variant="secondary")
                
                # Feedback visual para indicar quando o cache é atualizado
                with gr.Accordion("🔄 Status", open=False):
                    gr.Markdown("""
                    As estatísticas são atualizadas automaticamente quando:
                    - Uma nova consulta é processada
                    - O cache é limpo
                    - Uma resposta é recuperada do cache
                    """)
                
                # Seção de informações
                with gr.Accordion("ℹ️ Sobre o Sistema", open=False):
                    gr.Markdown("""
                    ### Como funciona
                    Este sistema utiliza uma arquitetura hierárquica de agentes especializados:
                    
                    - **Manager**: Coordena todos os outros agentes
                    - **Research**: Encontra filmes com base em critérios
                    - **Details**: Fornece informações detalhadas
                    - **Recommendation**: Sugere conteúdo similar
                    - **People**: Informações sobre atores, diretores, etc.
                    
                    O sistema otimiza o uso de recursos respondendo diretamente quando possível
                    ou delegando para especialistas quando necessário.
                    """)
    
    # Lógica de eventos com retorno de estatísticas atualizadas
    msg.submit(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time])
    submit_btn.click(process_message, [msg, chatbot], [chatbot, cache_stats, cache_time])
    clear_btn.click(clear_history_and_cache, None, [chatbot, chatbot, cache_stats, cache_time])

# Iniciar a interface
if __name__ == "__main__":
    # Instalar dependências necessárias automaticamente
    import subprocess
    import sys
    
    print("Verificando dependências necessárias...")
    try:
        import gradio
    except ImportError:
        print("Instalando Gradio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        print("Gradio instalado com sucesso!")
    
    print(f"Iniciando {SYSTEM_NAME}...")
    demo.launch(share=True, inbrowser=True)