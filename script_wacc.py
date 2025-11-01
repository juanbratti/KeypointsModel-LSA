import matplotlib.pyplot as plt

# Datos
beam_sizes = [1, 5, 8, 16, 24, 32]
word_acc = [1.3, 1.2, 1.5, 0.6, 0.6, 0.6]  # valores de tu tabla

# Crear figura
plt.figure(figsize=(8, 4.5))

# Graficar Word Accuracy
plt.plot(beam_sizes, word_acc, 'o-', color='tab:blue', label='Word Accuracy')

# Configuración del gráfico
plt.xlabel("Beam Width")
plt.ylabel("Word Accuracy (%)")
plt.title("Impacto del Beam Width en la Word Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Guardar gráfico
plt.savefig("bm_acc4.png", dpi=300, bbox_inches='tight')

# Mostrar gráfico (opcional)
plt.show()
