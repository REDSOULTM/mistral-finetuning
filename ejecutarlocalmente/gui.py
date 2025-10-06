import tkinter as tk
from tkinter import ttk
import threading
import time

class LoadingGUI:
    def __init__(self, on_loaded_callback, load_model_fn=None):
        self.root = tk.Tk()
        self.root.title("Transportes Miramar - Cargando modelo")
        self.root.geometry("500x180")
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=40)
        self.label = tk.Label(self.root, text="Cargando modelo... Por favor espera.", font=("Arial", 14))
        self.label.pack(pady=10)
        self.on_loaded_callback = on_loaded_callback
        self.load_model_fn = load_model_fn
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

    def start_loading(self):
        threading.Thread(target=self._real_loading, daemon=True).start()
        self.root.mainloop()

    def _real_loading(self):
        # Simula pasos de optimización y carga real del modelo
        steps = [
            ("Optimizando modelo...", 10),
            ("Cargando pesos cuantizados...", 30),
            ("Inicializando KV cache...", 50),
            ("Compilando kernels...", 70),
            ("Preparando chat...", 90),
            ("¡Modelo cargado! Iniciando chat...", 100)
        ]
        for msg, percent in steps:
            self.progress['value'] = percent
            self.label.config(text=f"{msg} {percent}%")
            self.root.update_idletasks()
            time.sleep(1.5)
        # Aquí se puede llamar a la función real de carga del modelo si se provee
        if self.load_model_fn:
            self.load_model_fn()
        self.root.after(10, self._close_and_start_chat)

    def _close_and_start_chat(self):
        self.root.destroy()
        self.on_loaded_callback()

class ChatGUI:
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Transportes Miramar - Chat")
        self.root.geometry("600x400")
        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        self.text = tk.Text(self.chat_frame, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 12))
        self.text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.entry = tk.Entry(self.root, font=("Arial", 12))
        self.entry.pack(fill=tk.X, padx=10, pady=5)
        self.entry.bind("<Return>", self.send_message)
        self._init_conversation()

    def _init_conversation(self):
        self._append_message("Cliente: Hola quiero cotizar\n\n({\"origen\": \"\", \"destino\": \"\", \"fecha\": \"\", \"hora\": \"\", \"regreso\": \"\", \"cantidad\": 0, \"comentario adicional\": \"\"})")
        self._append_message("Transporte Miramar: Hola, soy Transportes Miramar. Estoy aquí para ayudarte con tu traslado en Chile.\n¿Desde dónde inicia tu viaje y hacia dónde deseas llegar?")

    def _append_message(self, msg):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, msg + "\n")
        self.text.config(state=tk.DISABLED)
        self.text.see(tk.END)

    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if user_msg:
            self._append_message(f"Cliente: {user_msg}")
            # Aquí deberías llamar al backend LLM y mostrar la respuesta
            self._append_message("Transporte Miramar: [Respuesta generada por el modelo]")
            self.entry.delete(0, tk.END)

    def start(self):
        self.root.mainloop()

class ModelSelectorGUI:
    def __init__(self, choices):
        self.root = tk.Tk()
        self.root.title("Selecciona modelo a utilizar")
        self.root.geometry("400x200")
        self.selected = tk.StringVar(value=list(choices.keys())[0])
        self.choices = choices
        label = tk.Label(self.root, text="Selecciona el modelo a utilizar:", font=("Arial", 14))
        label.pack(pady=20)
        for key, (_, desc) in choices.items():
            rb = tk.Radiobutton(self.root, text=desc, variable=self.selected, value=key, font=("Arial", 12))
            rb.pack(anchor="w", padx=40)
        btn = tk.Button(self.root, text="Continuar", command=self._on_continue, font=("Arial", 12))
        btn.pack(pady=20)
        self._done = False
        self._result = None
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

    def _on_continue(self):
        self._result = self.choices[self.selected.get()][0]
        self._done = True
        self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return self._result

# Ejemplo de uso:
if __name__ == "__main__":
    def cargar_modelo():
        # Aquí puedes poner la llamada real a la carga del modelo LLM si lo deseas
        time.sleep(2)  # Simulación de carga real
    def iniciar_chat():
        chat = ChatGUI(parent=None)
        chat.start()
    gui = LoadingGUI(iniciar_chat, load_model_fn=cargar_modelo)
    gui.start_loading()
