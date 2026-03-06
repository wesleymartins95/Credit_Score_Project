import pandas as pd
import numpy as np

def calc_woe_iv(data, feature, target, bins):
    df = data.copy()
    # Criar os bins
    df['bin'] = pd.cut(df[feature], bins=bins, include_lowest=True)
    
    # Agrupar por bin
    grouped = df.groupby('bin')[target].agg(['count','sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['count']
    
    # Calcular WOE
    grouped['woe'] = np.log(
        (grouped['non_event'] / grouped['non_event'].sum()) /
        (grouped['sum'] / grouped['sum'].sum())
    )
    
    # Calcular IV
    grouped['iv'] = (
        (grouped['non_event'] / grouped['non_event'].sum()) -
        (grouped['sum'] / grouped['sum'].sum())
    ) * grouped['woe']
    
    iv_value = grouped['iv'].sum()
    return grouped.reset_index(), iv_value

