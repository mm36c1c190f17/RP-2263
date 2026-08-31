import pandas as pd
import sys

def add_experiment(filler_type, silane, amount, fp_before, ep_before, fp_after, ep_after, rho_before, rho_after, P_strength, manufacturer='Дубна', rubber_type='СКТВ-1'):
    """Добавление нового эксперимента в базу данных"""
    file_path = "data/raw/experimental_data.csv"
    
    # Создаем новую запись
    new_data = pd.DataFrame([{
        'filler_type': filler_type,
        'silane_content': silane,
        'filler_amount': amount,
        'fp_before': fp_before,
        'ep_before': ep_before,
        'fp_after': fp_after,
        'ep_after': ep_after,
        'rho_before': rho_before,
        'rho_after': rho_after,
        'P_strength': P_strength,
        'manufacturer': manufacturer,
        'rubber_type': rubber_type
    }])
    
    # Загружаем существующие данные
    try:
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_data], ignore_index=True)
    except FileNotFoundError:
        df = new_data
    
    # Сохраняем
    df.to_csv(file_path, index=False)
    print(f"✅ Добавлен эксперимент:")
    print(f"   {filler_type} | силан: {silane} phr | наполнитель: {amount} phr")
    print(f"   Всего записей: {len(df)}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 11:
        print("Использование:")
        print("python add_experiment.py <filler> <silane> <amount> <fp_before> <ep_before> <fp_after> <ep_after> <rho_before> <rho_after> <P_strength>")
        print("\nПример:")
        print("python add_experiment.py шпинель 5 40 7.2 370 4.1 95 4.1e13 6.5e12 80")
        sys.exit(1)
    
    add_experiment(
        filler_type=sys.argv[1],
        silane=float(sys.argv[2]),
        amount=float(sys.argv[3]),
        fp_before=float(sys.argv[4]),
        ep_before=float(sys.argv[5]),
        fp_after=float(sys.argv[6]),
        ep_after=float(sys.argv[7]),
        rho_before=float(sys.argv[8]),
        rho_after=float(sys.argv[9]),
        P_strength=float(sys.argv[10])
    )
    
    print("\n🔄 Для переобучения модели запустите: python train.py")
