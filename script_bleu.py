import matplotlib.pyplot as plt

# Datos
beam_sizes = [1, 5, 8, 16, 24, 32]

bleu1 = [11.3, 9.4, 10.5, 9.7, 8.3, 7.4]
bleu2 = [3.6, 3.0, 3.6, 3.3, 2.8, 2.5]
bleu3 = [1.2, 1.4, 1.7, 1.6, 1.4, 1.2]
bleu4 = [0.5, 0.8, 1.0, 0.9, 0.8, 0.6]

# Crear figura
plt.figure(figsize=(8, 4.5))

# Graficar líneas BLEU
plt.plot(beam_sizes, bleu1, 'o-', label='BLEU-1')
plt.plot(beam_sizes, bleu2, 's-', label='BLEU-2')
plt.plot(beam_sizes, bleu3, '^-', label='BLEU-3')
plt.plot(beam_sizes, bleu4, 'x-', label='BLEU-4')

# Configuración del gráfico
plt.xlabel("Beam Width")
plt.ylabel("BLEU Score (%)")
plt.title("Impacto del Beam Width en el rendimiento del modelo")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Guardar gráfico
plt.savefig("bw_bleu5.png", dpi=300, bbox_inches='tight')

# Mostrar gráfico (opcional)
plt.show()
