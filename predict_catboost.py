import pandas as pd
from catboost import CatBoostRegressor, Pool

targets = ['strength_initial', 'elongation_initial', 'strength_aged_240h_250C',
           'elongation_aged_240h_250C', 'strength_aged_72h_250C', 
           'elongation_aged_72h_250C', 'resistivity', 'permittivity', 
           'tan_delta', 'dielectric_strength', 'ceramic_strength']

models = {}
for target in targets:
    model = CatBoostRegressor()
    model.load_model(f'models/catboost_{target}.cbm')
    models[target] = model

def predict(filler_type, filler_content=30, temp=115, time=15):
    input_data = pd.DataFrame([{
        'base_type': 'VMQ',
        'base_hardness': 70,
        'base_manufacturer': 'Xiameter',
        'filler_type': filler_type,
        'filler_manufacturer': 'JSC_Vostochnye_Ogneupory',
        'filler_content': filler_content,
        'temp': temp,
        'time': time
    }])
    
    cat_features = ['base_type', 'base_manufacturer', 'filler_type', 'filler_manufacturer']
    pool = Pool(input_data, cat_features=cat_features)
    
    results = {}
    for name, model in models.items():
        results[name] = float(model.predict(pool)[0])
    
    return results

if __name__ == "__main__":
    import sys
    filler = sys.argv[1] if len(sys.argv) > 1 else 'MgAl2O4'
    content = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f'🔮 {filler} {content} phr')
    print('=' * 50)
    results = predict(filler, content)
    for key, value in results.items():
        print(f'{key}: {value:.2f}')
