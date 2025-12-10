"""
Script de diagnóstico para verificar el dataset antes de entrenar
Ejecuta este script para ver el estado de tus datos
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("🔍 DIAGNÓSTICO DEL DATASET")
print("=" * 70)

try:
    # Cargar dataset
    df = pd.read_csv('data/academic_performance_master.csv')
    print(f"\n✅ Dataset cargado correctamente")
    print(f"   Dimensiones: {df.shape} (filas x columnas)")
    
    # Mostrar primeras filas
    print(f"\n📊 Primeras 5 filas del dataset:")
    print(df.head())
    
    # Información de columnas
    print(f"\n📋 Columnas disponibles:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col} ({df[col].dtype})")
    
    # Verificar columnas necesarias
    print(f"\n🔍 Verificando columnas necesarias:")
    columnas_necesarias = ['Nota_final', 'Asistencia', 'Tareas_entregadas', 'Participacion']
    
    for col in columnas_necesarias:
        if col in df.columns:
            print(f"   ✅ {col}: ENCONTRADA")
            # Estadísticas básicas
            print(f"      - Valores no nulos: {df[col].notna().sum()}/{len(df)}")
            print(f"      - Rango: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"      - Media: {df[col].mean():.2f}")
        else:
            print(f"   ❌ {col}: NO ENCONTRADA")
    
    # Análisis de Nota_final
    if 'Nota_final' in df.columns:
        print(f"\n📈 Análisis detallado de Nota_final:")
        print(f"   Total de registros: {len(df)}")
        print(f"   Valores válidos: {df['Nota_final'].notna().sum()}")
        print(f"   Valores nulos: {df['Nota_final'].isna().sum()}")
        
        # Limpiar nulos
        df_limpio = df.dropna(subset=['Nota_final'])
        
        print(f"\n   Estadísticas de notas:")
        print(f"   - Mínima: {df_limpio['Nota_final'].min():.2f}")
        print(f"   - Máxima: {df_limpio['Nota_final'].max():.2f}")
        print(f"   - Media: {df_limpio['Nota_final'].mean():.2f}")
        print(f"   - Mediana: {df_limpio['Nota_final'].median():.2f}")
        print(f"   - Desviación estándar: {df_limpio['Nota_final'].std():.2f}")
        
        # Crear variable Aprobado
        df_limpio['Aprobado'] = (df_limpio['Nota_final'] >= 14).astype(int)
        
        print(f"\n   📊 Distribución de Aprobado/Reprobado (umbral >= 14):")
        distribucion = df_limpio['Aprobado'].value_counts().sort_index()
        
        for clase in [0, 1]:
            if clase in distribucion.index:
                count = distribucion[clase]
                etiqueta = "Aprobados (1)" if clase == 1 else "Reprobados (0)"
                print(f"   - {etiqueta}: {count} estudiantes ({count/len(df_limpio)*100:.1f}%)")
            else:
                etiqueta = "Aprobados (1)" if clase == 1 else "Reprobados (0)"
                print(f"   - {etiqueta}: 0 estudiantes (0.0%)")
        
        # VERIFICACIÓN CRÍTICA
        print(f"\n{'='*70}")
        if len(distribucion) < 2:
            print("❌ ¡ERROR CRÍTICO!")
            print(f"   Solo hay estudiantes {'APROBADOS' if distribucion.index[0] == 1 else 'REPROBADOS'}")
            print(f"\n💡 POSIBLES CAUSAS:")
            print(f"   1. Todas las notas son >= 14 (todos aprobados)")
            print(f"   2. Todas las notas son < 14 (todos reprobados)")
            print(f"\n🔧 SOLUCIONES:")
            print(f"   1. Verifica que tu CSV tenga notas variadas")
            print(f"   2. Ajusta el umbral de aprobación")
            print(f"   3. Revisa los valores de la columna 'Nota_final'")
        else:
            print("✅ ¡DATASET VÁLIDO!")
            print(f"   Hay datos de ambas clases (Aprobados y Reprobados)")
            
            # Verificar balance
            min_class = distribucion.min()
            max_class = distribucion.max()
            balance = min_class / max_class
            
            print(f"\n   Balance de clases: {balance:.2%}")
            if balance < 0.2:
                print(f"   ⚠️  Las clases están muy desbalanceadas")
                print(f"   Se recomienda al menos 20% de la clase minoritaria")
            elif balance < 0.5:
                print(f"   ⚠️  Las clases están algo desbalanceadas")
            else:
                print(f"   ✅ Balance aceptable")
            
            # Muestras mínimas
            if min_class < 5:
                print(f"\n   ⚠️  La clase minoritaria tiene solo {min_class} muestras")
                print(f"   Se recomienda al menos 10 muestras por clase")
            else:
                print(f"\n   ✅ Suficientes muestras por clase ({min_class} mínimo)")
        
        print(f"{'='*70}")
        
        # Distribución por rangos de notas
        print(f"\n📊 Distribución por rangos de notas:")
        bins = [0, 7, 11, 14, 17, 21]
        labels = ['0-7 (Muy bajo)', '7-11 (Bajo)', '11-14 (Regular)', '14-17 (Bueno)', '17-20 (Excelente)']
        df_limpio['Rango'] = pd.cut(df_limpio['Nota_final'], bins=bins, labels=labels, include_lowest=True)
        
        for rango, count in df_limpio['Rango'].value_counts().sort_index().items():
            print(f"   {rango}: {count} estudiantes ({count/len(df_limpio)*100:.1f}%)")
    
    else:
        print(f"\n❌ ERROR: No se encontró la columna 'Nota_final'")
    
    # Verificar valores nulos en otras columnas
    print(f"\n🔍 Valores nulos por columna:")
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        for col, count in nulos[nulos > 0].items():
            print(f"   ⚠️  {col}: {count} nulos ({count/len(df)*100:.1f}%)")
    else:
        print(f"   ✅ No hay valores nulos")
    
    print(f"\n{'='*70}")
    print("✅ DIAGNÓSTICO COMPLETADO")
    print(f"{'='*70}")

except FileNotFoundError:
    print(f"\n❌ ERROR: No se encontró el archivo 'data/academic_performance_master.csv'")
    print(f"\n📁 Verifica que:")
    print(f"   1. El archivo existe")
    print(f"   2. Está en la carpeta 'data/'")
    print(f"   3. El nombre es correcto (case sensitive)")

except Exception as e:
    print(f"\n❌ ERROR INESPERADO: {str(e)}")
    import traceback
    traceback.print_exc()