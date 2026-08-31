import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv('data/raw/experimental_data.csv')

print("=" * 60)
print("СТАТИСТИКА ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ")
print("=" * 60)

print(f"\n📊 Всего экспериментов: {len(df)}")
print(f"\n📋 Наполнители:")
print(df['filler_type'].value_counts())

print(f"\n📈 Статистика свойств:")
print(df[['fp_before', 'ep_before', 'fp_after', 'ep_after', 'P_strength']].describe())

# Сохраняем графики
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df.boxplot(column='fp_before', by='filler_type', ax=axes[0,0])
axes[0,0].set_title('Прочность до старения')
df.boxplot(column='ep_before', by='filler_type', ax=axes[0,1])
axes[0,1].set_title('Удлинение до старения')
df.boxplot(column='fp_after', by='filler_type', ax=axes[1,0])
axes[1,0].set_title('Прочность после старения')
df.boxplot(column='P_strength', by='filler_type', ax=axes[1,1])
axes[1,1].set_title('Прочность остатка')

plt.tight_layout()
plt.savefig('data_statistics.png', dpi=100)
print("\n✅ График сохранен как data_statistics.png")
