#!/usr/bin/env python3
"""Script para aplicar optimizaciones críticas de prompts."""

import re

def optimize_vendedor_py():
    """Aplicar optimizaciones críticas al archivo vendedor.py"""
    
    with open("vendedor.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Patrón para encontrar el método validate_input_llm completo
    pattern = r'(def validate_input_llm\(self, user_input\):.*?return \{\})\s*return \{\}'
    
    # Nuevo método optimizado
    new_method = '''def validate_input_llm(self, user_input):
        if self.model is None or self.tokenizer is None:
            return {}
        
        # PROMPT ULTRA-COMPACTO para máxima velocidad (reducido de ~1500 a ~50 tokens)
        prompt = f"""Extraer datos de: "{user_input}"
JSON campos: nombre, rut, correo, origen, destino, fecha (YYYY-MM-DD), hora (HH:MM), regreso (sí/no), cantidad, comentario adicional.
Estado: {json.dumps(self.state, ensure_ascii=False)}
JSON:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # REDUCIDO para velocidad
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )'''
    
    # Buscar y reemplazar el método completo
    if "def validate_input_llm" in content:
        print("🔧 Aplicando optimización de prompt...")
        
        # Buscar el inicio del método
        start_pos = content.find("def validate_input_llm(self, user_input):")
        if start_pos == -1:
            print("❌ No se encontró el método validate_input_llm")
            return
            
        # Buscar el final del método (siguiente def o final del archivo)
        remaining = content[start_pos:]
        end_pattern = r'\n    def '
        match = re.search(end_pattern, remaining)
        
        if match:
            end_pos = start_pos + match.start()
            method_content = content[start_pos:end_pos]
        else:
            # Si no hay siguiente método, buscar el final de la clase
            class_end = remaining.find("\n\nclass ")
            if class_end != -1:
                end_pos = start_pos + class_end
            else:
                end_pos = len(content)
            method_content = content[start_pos:end_pos]
        
        print(f"📏 Método original: {len(method_content)} caracteres")
        print(f"📏 Método optimizado: {len(new_method)} caracteres")
        print(f"🚀 Reducción: {len(method_content) / len(new_method):.1f}x menor")
        
        # Hacer el reemplazo
        new_content = content[:start_pos] + new_method + content[end_pos:]
        
        # Guardar el archivo optimizado
        with open("vendedor.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print("✅ Optimización aplicada exitosamente")
    else:
        print("❌ No se encontró el método validate_input_llm")

if __name__ == "__main__":
    optimize_vendedor_py()
