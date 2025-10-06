#!/usr/bin/env python3
"""
VerificarTodo.py - Verificador rápido de dependencias

Este script verifica si tienes todo lo necesario para ejecutar
el proyecto de fine-tuning de Mistral organizado en carpetas.

Estructura del proyecto:
- Instalar Requesitos/ (este verificador)
- RealizarFineTuning/ (script principal)
- Dataset de Miramar/ (datasets personalizados)
"""

import sys
import subprocess
import os
import importlib

def print_header():
    print("🔍" + "=" * 60 + "🔍")
    print("   VERIFICADOR DE DEPENDENCIAS PARA MISTRAL")
    print("     Proyecto organizado con carpetas")
    print("🔍" + "=" * 60 + "🔍")

def check_import(module_name, display_name, critical=True):
    """Verifica si un módulo se puede importar"""
    try:
        if module_name == "unsloth_special":
            # Verificación especial para Unsloth
            import unsloth
            from unsloth import FastLanguageModel
            print(f"✅ {display_name}: OK")
            return True
        elif module_name == "awq_special":
            import awq  # type: ignore
            print(f"✅ {display_name}: OK")
            return True
        else:
            importlib.import_module(module_name)
            print(f"✅ {display_name}: OK")
            return True
    except ImportError:
        if critical:
            print(f"❌ {display_name}: NO INSTALADO")
        else:
            print(f"⚠️  {display_name}: No encontrado (opcional)")
        return False
    except Exception as e:
        print(f"🔶 {display_name}: Error ({str(e)[:30]}...)")
        return False

def check_gpu():
    """Verifica GPU"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU NVIDIA: Detectada")
            return True
        else:
            print("❌ GPU NVIDIA: No detectada")
            return False
    except:
        print("❌ GPU NVIDIA: No disponible")
        return False

def check_cuda_pytorch():
    """Verifica CUDA en PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA en PyTorch: OK ({gpu_name})")
            return True
        else:
            print("❌ CUDA en PyTorch: No disponible")
            return False
    except:
        print("❌ PyTorch: No instalado")
        return False

def main():
    print_header()
    print("\n📋 Verificando dependencias críticas...")
    
    # Dependencias críticas
    critical_deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("sentencepiece", "SentencePiece"),
        ("bitsandbytes", "BitsAndBytes"),
        ("awq_special", "AutoAWQ"),
        ("unsloth_special", "Unsloth"),
        ("unsloth_zoo", "Unsloth Zoo"),
    ]
    
    missing = []
    for module, name in critical_deps:
        if not check_import(module, name):
            missing.append(name)
    
    # Verificar GPU
    print("\n🖥️  Verificando hardware...")
    gpu_ok = check_gpu()
    cuda_ok = check_cuda_pytorch()
    # Verificar estructura del proyecto
    print("\n📁 VERIFICANDO ESTRUCTURA DEL PROYECTO:")
    
    # Verificar carpetas principales
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    folders_to_check = [
        ("RealizarFineTuning", "Scripts de entrenamiento"),
        ("Dataset_de_Miramar", "Datasets personalizados"),  # Updated folder name
        ("Instalar Requesitos", "Scripts de instalación")
    ]
    
    structure_ok = True
    for folder, description in folders_to_check:
        folder_path = os.path.join(project_root, folder)
        if os.path.exists(folder_path):
            print(f"✅ {folder}/ - {description}")
        else:
            print(f"❌ {folder}/ - {description} (NO ENCONTRADA)")
            structure_ok = False
    
    # Verificar archivo principal
    main_script = os.path.join(project_root, "RealizarFineTuning", "mistral_finetuning_final.py")
    if os.path.exists(main_script):
        print("✅ mistral_finetuning_final.py - Script principal encontrado")
    else:
        print("❌ mistral_finetuning_final.py - Script principal NO encontrado")
        structure_ok = False
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN:")
    
    if not missing and cuda_ok and structure_ok:
        print("🎉 ¡TODO LISTO! Puedes ejecutar:")
        print("   cd ../RealizarFineTuning")
        print("   python mistral_finetuning_final.py")
    elif not missing and gpu_ok and structure_ok:
        print("✅ Dependencias y estructura OK, pero CUDA no disponible")
        print("⚠️  El entrenamiento será MUY lento")
        print("💡 Ejecuta: python InstalarTodo.py")
    elif missing or not structure_ok:
        print("❌ Problemas encontrados:")
        if missing:
            print("   Dependencias faltantes:")
            for dep in missing:
                print(f"   • {dep}")
        if not structure_ok:
            print("   • Estructura del proyecto incompleta")
        print("\n🚀 SOLUCIÓN: Ejecuta el instalador automático:")
        print("   python3 InstalarTodo.py")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
