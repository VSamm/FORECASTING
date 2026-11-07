from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "inferencia_df_transformado.csv"
MODEL_PATH = ROOT_DIR / "models" / "models" / "modelo_final.joblib"
LAG_COLUMNS = [f"unidades_vendidas_lag_{number}" for number in range(1, 8)]
MOVING_AVERAGE_COLUMN = "media_movil_7_dias"
SCENARIOS = {
	"Actual (0%)": 0.0,
	"Competencia -5%": -0.05,
	"Competencia +5%": 0.05,
}
WEEKDAYS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


st.set_page_config(
	page_title="Forecast | Noviembre 2025",
	page_icon="📈",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
	:root {
		--ink: #172a3a;
		--muted: #647783;
		--line: #dce5e8;
		--paper: #f5f8f6;
		--green: #0c8b72;
		--blue: #2874a6;
		--coral: #e85d4a;
		--gold: #d89c25;
	}
	html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
	.stApp { background: radial-gradient(ellipse at 92% 0%, #e8f3f0 0, transparent 34%), var(--paper); color: var(--ink); }
	h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: 0; }
	h1 { font-size: 2rem !important; font-weight: 800 !important; }
	[data-testid="stSidebar"] { background: #eaf1ef; border-right: 1px solid var(--line); }
	[data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line); border-top: 3px solid var(--green); padding: 16px 18px; border-radius: 6px; min-height: 112px; }
	[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
	[data-testid="stMetricValue"] { color: var(--ink) !important; font-family: 'Manrope', sans-serif; }
	[data-testid="stAppViewContainer"] p,
	[data-testid="stSidebar"] p,
	[data-testid="stWidgetLabel"] p,
	[data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
	div.stButton > button { background: var(--green); color: white; border: 0; border-radius: 6px; min-height: 48px; font-weight: 700; }
	div.stButton > button:hover { background: #08745f; color: white; border: 0; }
	hr { border-color: var(--line); }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
	data = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
	if data.empty:
		raise ValueError("El archivo de inferencia no contiene registros.")
	return data


@st.cache_resource(show_spinner=False)
def load_model():
	return joblib.load(MODEL_PATH)


def format_eur(value):
	return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def simulate_product(product_data, model, features, discount_adjustment, competition_adjustment):
	product_data = product_data.sort_values("fecha").copy().reset_index(drop=True)
	initial_lags = product_data.loc[0, LAG_COLUMNS]
	if initial_lags.isna().any():
		missing_lags = initial_lags[initial_lags.isna()].index.tolist()
		raise ValueError(f"Faltan lags iniciales del 1 de noviembre: {', '.join(missing_lags)}")

	product_data["precio_venta"] = product_data["precio_base"] * (1 - discount_adjustment / 100)
	product_data["descuento_porcentaje"] = discount_adjustment
	product_data["precio_competencia"] = product_data["precio_competencia"] * (1 + competition_adjustment)
	product_data["ratio_precio"] = np.where(
		product_data["precio_competencia"] != 0,
		product_data["precio_venta"] / product_data["precio_competencia"],
		np.nan,
	)

	lag_state = {column: float(initial_lags[column]) for column in LAG_COLUMNS}
	predictions = []
	for day_index in range(len(product_data)):
		row = product_data.loc[[day_index]].copy()
		if day_index > 0:
			for column, value in lag_state.items():
				row.loc[:, column] = value
			row.loc[:, MOVING_AVERAGE_COLUMN] = float(np.mean(list(lag_state.values())))

		missing_features = [feature for feature in features if feature not in row.columns]
		if missing_features:
			raise ValueError(f"El modelo requiere columnas que no existen en los datos: {', '.join(missing_features)}")
		feature_row = row.loc[:, features]
		if feature_row.isna().any(axis=None):
			missing = feature_row.columns[feature_row.isna().any()].tolist()
			raise ValueError(f"Hay valores vacíos en las variables del modelo: {', '.join(missing)}")

		predicted_units = max(0.0, float(model.predict(feature_row)[0]))
		predictions.append(predicted_units)
		lag_state = {
			LAG_COLUMNS[0]: predicted_units,
			**{LAG_COLUMNS[position]: lag_state[LAG_COLUMNS[position - 1]] for position in range(1, 7)},
		}

	product_data["unidades_predichas"] = predictions
	product_data["ingresos_proyectados"] = product_data["unidades_predichas"] * product_data["precio_venta"]
	return product_data


try:
	data = load_data()
	model = load_model()
except Exception as error:
	st.error(f"No se pudieron cargar los datos o el modelo: {error}")
	st.stop()

required_columns = [
	"fecha", "nombre", "precio_base", "precio_competencia", "precio_venta",
	"descuento_porcentaje", MOVING_AVERAGE_COLUMN, *LAG_COLUMNS,
]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
	st.error(f"Faltan columnas requeridas en el CSV: {', '.join(missing_columns)}")
	st.stop()

if not hasattr(model, "feature_names_in_"):
	st.error("El modelo no expone feature_names_in_; no se puede verificar el orden de entrada.")
	st.stop()
MODEL_FEATURES = list(model.feature_names_in_)

products = sorted(data["nombre"].dropna().unique().tolist())
if not products:
	st.error("No se encontraron productos en el archivo de inferencia.")
	st.stop()

with st.sidebar:
	st.title("🎛️ Controles de Simulación")
	with st.form("simulation_controls"):
		selected_product = st.selectbox("Producto", products)
		discount_adjustment = st.slider("Ajuste de descuento", -50, 50, 0, 5, format="%d%%")
		selected_scenario = st.radio("Escenario de competencia", list(SCENARIOS.keys()))
		simulate_clicked = st.form_submit_button("🚀 Simular Ventas", use_container_width=True)

if simulate_clicked:
	try:
		selected_data = data.loc[data["nombre"].eq(selected_product)].copy()
		if selected_data.empty:
			raise ValueError("No hay registros para el producto seleccionado.")
		if len(selected_data) != 30 or selected_data["fecha"].dt.month.ne(11).any():
			raise ValueError("Se esperaban exactamente 30 registros de noviembre para el producto seleccionado.")

		with st.spinner("Calculando la predicción recursiva y los escenarios..."):
			scenario_forecasts = {
				scenario: simulate_product(selected_data, model, MODEL_FEATURES, discount_adjustment, adjustment)
				for scenario, adjustment in SCENARIOS.items()
			}
			selected_forecast = scenario_forecasts[selected_scenario]
		st.session_state["forecast_result"] = {
			"product": selected_product,
			"scenario": selected_scenario,
			"discount": discount_adjustment,
			"selected": selected_forecast,
			"scenarios": scenario_forecasts,
		}
	except Exception as error:
		st.error(f"No se pudo completar la simulación: {error}")

result = st.session_state.get("forecast_result")
if result is None:
	st.title("📈 Simulación de ventas | Noviembre 2025")
	st.info("Selecciona un producto y configura el escenario en el panel lateral para comenzar.")
	st.stop()

forecast = result["selected"]
st.title("📈 Simulación de ventas | Noviembre 2025")
st.subheader(f"{result['product']}  ·  {result['scenario']}  ·  Descuento {result['discount']:+d}%")
st.caption("Proyección recursiva diaria del 1 al 30 de noviembre. Los lags iniciales del primer día se conservan desde el archivo.")

if not {"precio_competencia", "ratio_precio"}.intersection(MODEL_FEATURES):
	st.warning(
		"El artefacto actual no usa precio de competencia ni variables de calendario. "
		"Los escenarios ajustan el precio de referencia mostrado, pero no pueden cambiar las predicciones; "
		"el modelo tampoco aplica un incremento especial por Black Friday."
	)

total_units = float(forecast["unidades_predichas"].sum())
total_revenue = float(forecast["ingresos_proyectados"].sum())
average_price = float(forecast["precio_venta"].mean())
average_discount = float(forecast["descuento_porcentaje"].mean())

kpi_columns = st.columns(4)
kpi_columns[0].metric("Unidades proyectadas", f"{total_units:,.0f}".replace(",", "."))
kpi_columns[1].metric("Ingresos proyectados", format_eur(total_revenue))
kpi_columns[2].metric("Precio medio", format_eur(average_price))
kpi_columns[3].metric("Descuento medio", f"{average_discount:.0f}%")

st.divider()
st.subheader("📊 Predicción diaria")
plot_data = forecast.assign(dia=forecast["fecha"].dt.day)
black_friday = forecast.loc[forecast["fecha"].dt.day.eq(28)].iloc[0]
chart_values = [
	{"dia": int(row.dia), "unidades_predichas": float(row.unidades_predichas)}
	for row in plot_data[["dia", "unidades_predichas"]].itertuples(index=False)
]
black_friday_value = float(black_friday["unidades_predichas"])
chart_spec = {
	"$schema": "https://vega.github.io/schema/vega-lite/v6.json",
	"background": "#ffffff",
	"width": "container",
	"height": 300,
	"layer": [
		{
			"data": {"values": chart_values},
			"mark": {"type": "line", "color": sns.color_palette(["#0c8b72"]).as_hex()[0], "strokeWidth": 3},
			"encoding": {
				"x": {
					"field": "dia",
					"type": "quantitative",
					"scale": {"domain": [1, 30], "zero": False, "nice": False},
					"axis": {
						"title": "Día de noviembre",
						"values": list(range(1, 31)),
						"labelAngle": 0,
						"labelOverlap": False,
					},
				},
				"y": {
					"field": "unidades_predichas",
					"type": "quantitative",
					"axis": {"title": "Unidades predichas"},
				},
			},
		},
		{
			"data": {"values": [{"dia": 28}]},
			"mark": {"type": "rule", "color": "#e85d4a", "strokeDash": [5, 4], "strokeWidth": 2},
			"encoding": {"x": {"field": "dia", "type": "quantitative"}},
		},
		{
			"data": {"values": [{"dia": 28, "unidades_predichas": black_friday_value}]},
			"mark": {"type": "point", "filled": True, "color": "#e85d4a", "size": 130, "stroke": "white", "strokeWidth": 1.5},
			"encoding": {
				"x": {"field": "dia", "type": "quantitative"},
				"y": {"field": "unidades_predichas", "type": "quantitative"},
			},
		},
		{
			"data": {"values": [{"dia": 28, "unidades_predichas": black_friday_value, "etiqueta": "Black Friday · 28 nov"}]},
			"mark": {"type": "text", "align": "right", "dx": -8, "dy": -15, "fontWeight": "bold", "color": "#b84234"},
			"encoding": {
				"x": {"field": "dia", "type": "quantitative"},
				"y": {"field": "unidades_predichas", "type": "quantitative"},
				"text": {"field": "etiqueta"},
			},
		},
	],
	"config": {
		"view": {"stroke": None},
		"axis": {
			"labelColor": "#172a3a",
			"titleColor": "#172a3a",
			"gridColor": "#e7eeee",
			"domainColor": "#dce5e8",
			"tickColor": "#dce5e8",
		},
	},
}
st.vega_lite_chart(chart_spec, width="stretch")

st.divider()
st.subheader("🗓️ Detalle por día")
table_data = pd.DataFrame(
	{
		"Fecha": forecast["fecha"].dt.strftime("%d/%m/%Y"),
		"Día de la semana": forecast["fecha"].dt.day_of_week.map(lambda day: WEEKDAYS_ES[day]),
		"Precio de venta": forecast["precio_venta"],
		"Precio competencia": forecast["precio_competencia"],
		"Descuento aplicado": forecast["descuento_porcentaje"].map(lambda value: f"{value:.0f}%"),
		"Unidades predichas": forecast["unidades_predichas"],
		"Ingresos proyectados": forecast["ingresos_proyectados"],
	}
)
black_friday_row = forecast["fecha"].dt.day.eq(28).to_numpy()
styled_table = table_data.style.format(
	{
		"Precio de venta": lambda value: format_eur(value),
		"Precio competencia": lambda value: format_eur(value),
		"Unidades predichas": lambda value: f"{value:,.0f}".replace(",", "."),
		"Ingresos proyectados": lambda value: format_eur(value),
	}
).apply(
	lambda row: ["background-color: #fff0ed; color: #8d352b; font-weight: 700" if black_friday_row[row.name] else ""] * len(row),
	axis=1,
)
table_data.loc[black_friday_row, "Día de la semana"] = "🛍️ Black Friday"
st.dataframe(styled_table, width="stretch", hide_index=True)

st.divider()
st.subheader("🧭 Comparativa de escenarios de competencia")
scenario_columns = st.columns(3)
for column, (scenario, scenario_data) in zip(scenario_columns, result["scenarios"].items()):
	column.metric(
		scenario,
		f"{scenario_data['unidades_predichas'].sum():,.0f}".replace(",", ".") + " unidades",
	)
	column.caption(f"Ingresos proyectados: {format_eur(float(scenario_data['ingresos_proyectados'].sum()))}")
