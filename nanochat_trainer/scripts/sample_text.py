verification_text = r"""Hello, world! 🌍🚀  
Café naïve – résumé, coöperate, fiancé.  
数学, русский, العربية, हिन्दी, 日本語, emojis 😄👍💡  
Symbols: © ™ ∆ ∑ √ ≈ • ¶ § ±  
Newlines, tabs\tand multiple   spaces.
""".strip()

general_text = r"""
Artificial intelligence (AI) is the capability of computational systems to perform 
tasks typically associated with human intelligence, such as learning, reasoning, 
problem-solving, perception, and decision-making. It is a field of research in computer 
science that develops and studies methods and software that enable machines to perceive 
their environment and use learning and intelligence to take actions that maximize their 
chances of achieving defined goals.\n

High-profile applications of AI include advanced web search engines (e.g., Google Search); 
recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants 
(e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative 
tools (e.g., language models and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, 
many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, 
often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."
""".strip()

code_text = r"""
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, weight_decay=0.0):
        if isinstance(parameters, (list, tuple)):
            if isinstance(parameters[0], dict):
                self.param_groups = []
                for group in parameters:
                    param_group = {
                        "params": [p for p in group["params"] if p.requires_grad],
                        "lr": group.get("lr", lr),
                        "weight_decay": group.get("weight_decay", weight_decay)
                    }    
                    self.param_groups.append(param_group)
            else:
                self.param_groups = [{
                    'params': [p for p in parameters if p.requires_grad],
                    'lr': lr,
                    'weight_decay': weight_decay,
                }]
        else:
            self.param_groups = [{
                'params': [p for p in parameters if p.requires_grad],
                'lr': lr,
                'weight_decay': weight_decay,
            }]

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    g = p.grad
                    if weight_decay != 0.0:
                        g = g + weight_decay * p.data
                    p.data -= lr * g
""".strip()

japanese_text = r"""
LLM構築用の日本語インストラクション(チャット)データセット

主に，英語で構築されたLLMモデルなどに対して，チャット(Instruction)応答タスクに関してLoRAなどでチューニングするために使用できます．

※様々な公開言語資源を利用させていただきました．関係各位にはこの場を借りて御礼申し上げます．
""".strip()

