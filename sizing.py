import pandas as pd
import math
from datetime import datetime

def sizing(user_data):
    df_irradiation = pd.read_csv(r"D:\===MyFiles===\TCC\BASES DE DADOS\IRRADIAÇÃO SOLAR\global_horizontal_means_sedes-munic.csv", sep=';')
    df_modules = pd.read_excel(r"D:\===MyFiles===\TCC\BASES DE DADOS\INVERSORES_MÓDULOS\DATABASE.xlsx", sheet_name='MODULES')
    df_inverters = pd.read_excel(r"D:\===MyFiles===\TCC\BASES DE DADOS\INVERSORES_MÓDULOS\DATABASE.xlsx", sheet_name='INVERTERS')
    df_tarifas = pd.read_excel(r"D:\===MyFiles===\TCC\BASES DE DADOS\TARIFAS_B1_ANEEL_24.xlsx", dtype='str')

    df_tarifas = df_tarifas.sort_values('Tarifa Convencional', ascending=True).drop_duplicates(subset='UF')
    df_tarifas['Tarifa Convencional'] = df_tarifas['Tarifa Convencional'].astype(float)

    # CONSUMO ENERGÉTICO MENSAL EM kWh/mês
    energy_tax = df_tarifas.loc[df_tarifas['UF'] == user_data[0]['uf'], 'Tarifa Convencional'].values[0]
    user_energy_bill = user_data[1]
    taxa_minima = 50 
    average_consumption_month = (user_energy_bill/energy_tax) - taxa_minima # kWh/mês

    # LOCALIZAÇÃO
    city_name = user_data[0]['localidade']
    local_temperature = 35 # °C
    residencial_type = user_data[2]

    # PRODUÇÃO DESEJADA
    pv_percentage_generated = user_data[3]/100 # 0% a 100%
    budget = user_data[4]

    energia_geracao = average_consumption_month/30
    energia_geracao = energia_geracao * pv_percentage_generated

    city_irradiance_data = df_irradiation[df_irradiation['NAME'] == city_name]
    irradiance_columns = city_irradiance_data.iloc[:, 7:18]
    irradiance_local = irradiance_columns.mean().mean() # Wh/(m^2*dia)
    irradiation_stc = 1000 # W/m^2
    tempo_exposicao = irradiance_local / irradiation_stc # h/dia

    derating = 0.74

    pv_installed_capacity = (energia_geracao/(tempo_exposicao * derating)) * 1000 # Wp

    sizing_modules = pd.DataFrame()

    for index, row in df_modules.iterrows():
        pot_max_nominal = row['POT. MAX (W)']
        temperature_coef = row['COEF. TEMPERATURA PMAX (%)']
        
        module_area = (row['COMPRIMENTO (mm)']/1000 * row['LARGURA (mm)']/1000) # m2
        
        standard_temperature = 25
        actual_module_max_pot = pot_max_nominal * (1 + (local_temperature - standard_temperature) * (temperature_coef/100))  # W
        
        modules_quantity = math.ceil(pv_installed_capacity / actual_module_max_pot)
        
        result_row = pd.DataFrame({
        'MARCA': row['MARCA'],
        'MODELO': row['MODELO'],
        'QUANTIDADE DE MÓDULOS': modules_quantity,
        'ÁREA TOTAL (m2)': module_area * modules_quantity,
        'CUSTO TOTAL (R$)': row['VALOR (R$)'] * modules_quantity,
        'POTÊNCIA TOTAL (W)': row['POT. MAX (W)'] * modules_quantity,
        'TENSÃO CIRCUITO ABERTO (V)': row['TENSÃO CIRCUITO ABERTO (V)'],
        'TENSÃO OPERAÇÃO (V)': row['TENSÃO OPERAÇÃO (V)'],
        'CORRENTE C.C (A)': row['CORRENTE C.C (A)'],
        'CORRENTE OPERAÇÃO (A)': row['CORRENTE OPERAÇÃO (A)']
        }, index=[index])
        
        sizing_modules = pd.concat([sizing_modules, result_row], ignore_index=True)


    # DIMENSIONAMENTO DOS INVERSORES
    sizing_inverters = pd.DataFrame()

    for index, row_module in sizing_modules.iterrows():
        for index, row_inverter in df_inverters.iterrows():
            qtde_string = row_inverter['QTD_MPPT'] * row_inverter['STRING_MPPT']

            for i in range(qtde_string):
                qtde_modulos_string = row_module['QUANTIDADE DE MÓDULOS'] / qtde_string
                
                total_system_voltage = row_module['TENSÃO CIRCUITO ABERTO (V)'] * qtde_modulos_string
                total_operating_system_voltage = row_module['TENSÃO OPERAÇÃO (V)'] * qtde_modulos_string
            
                total_system_current = row_module['CORRENTE C.C (A)'] * qtde_string
                total_operating_current = row_module['CORRENTE OPERAÇÃO (A)'] * qtde_string

                total_system_power = total_system_voltage * total_system_current

                df_inverters_sized = df_inverters[
                    (df_inverters['IN_MAX_POWER'].astype(float) >= total_system_power)
                ]

                sizing_inverters = pd.concat([sizing_inverters, df_inverters_sized], ignore_index=True)

    sizing_inverters.drop_duplicates()

    lower_cost = float('inf')
    best_module_row = None
    best_inverter_row = None

    for _, row_module in sizing_modules.iterrows():
        for _, row_inverter in df_inverters.iterrows():
            total_cost = row_module['CUSTO TOTAL (R$)'] + row_inverter['PRICE']

            if total_cost < lower_cost:
                lower_cost = total_cost
                best_module_row = row_module
                best_inverter_row = row_inverter

    print("Menor custo total encontrado:", lower_cost)
    print("Módulo correspondente:")
    print(best_module_row)
    print("Inversor correspondente:")
    print(best_inverter_row)

    # Variáveis
    total_investment = lower_cost  # Investimento total do sistema
    generation_year = (pv_installed_capacity * 365)/1000  # Geração de energia anual em kWh
    electricity_tariff = energy_tax  # Tarifa de energia elétrica em R$ por kWh
    escalation_rate = 0.05  # Crescimento % da tarifa de energia por ano
    maintenance_cost = 0.01 * total_investment  # Custo de manutenção anual da planta
    generation_degradation = 0.007  # Redução da produção por ano 
    minimum_tariff_year = 12 * 50 * electricity_tariff  # Minimo pago mensalmente para a concessionária

    # Cálculo do fluxo de caixa líquido anual
    years = 0
    cumulative_cashflow = -total_investment

    while cumulative_cashflow < 0:
        years += 1
        
        # Geração de energia no ano atual considerando a degradação
        generation_year -= generation_year * generation_degradation
        
        # Tarifa de eletricidade no ano atual considerando a escalada
        electricity_tariff += electricity_tariff * escalation_rate

        # # Receita anual
        annual_revenue = generation_year * electricity_tariff

        # # Fluxo de caixa líquido anual
        annual_cashflow = annual_revenue - maintenance_cost - minimum_tariff_year
        cumulative_cashflow += annual_cashflow

    # Cálculo do payback
    payback_years = years - 1 + (-cumulative_cashflow / annual_cashflow)

    messages = []

    messages.append("Com base nas informações fornecidas, encontramos a melhor solução para o seu sistema de energia solar fotovoltaica:")
    
    messages.append("==MÓDULO RECOMENDADO==")
    messages.append(f"Marca: {best_module_row['MARCA']}, Modelo: {best_module_row['MODELO']}")
    messages.append(f"Quantidade de módulos: {best_module_row['QUANTIDADE DE MÓDULOS']}")
    messages.append(f"Área total necessária: {round(best_module_row['ÁREA TOTAL (m2)'], 2)} m²")
    
    messages.append("==INVERSOR RECOMENDADO==")
    messages.append(f"Marca: {best_inverter_row['MARCA']}, Modelo: {best_inverter_row['MODELO']}")
    
    messages.append("==PAYBACK==")
    messages.append(f"Considerando o investimento total de aproximadamente R$ {total_investment}, o payback estimado para o projeto é de aproximadamente {round(payback_years, 2)} anos.")
    
    messages.append("Ficamos à disposição para qualquer dúvida ou para discutir as opções em mais detalhes.")
    
    return messages