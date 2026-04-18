-- =============================================================================
-- DDL-схема базы данных ИС прогнозирования пассажиропотока
-- СУБД: PostgreSQL 14+
-- Нормализация: 3НФ
-- Приложение А к ВКР «Прогнозирование пассажиропотока на междугородних рейсах»
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Справочники
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS Region (
    region_id   SERIAL       PRIMARY KEY,
    name        VARCHAR(150) NOT NULL,
    federal_district VARCHAR(100),
    population  INTEGER,
    area_km2    NUMERIC(12,2),
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS Station (
    station_id  SERIAL       PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    region_id   INTEGER      NOT NULL REFERENCES Region(region_id),
    address     TEXT,
    latitude    NUMERIC(9,6),
    longitude   NUMERIC(9,6),
    is_terminal BOOLEAN      DEFAULT FALSE,
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS Route (
    route_id        SERIAL       PRIMARY KEY,
    route_code      VARCHAR(50)  NOT NULL UNIQUE,
    name            VARCHAR(300) NOT NULL,
    origin_id       INTEGER      NOT NULL REFERENCES Station(station_id),
    destination_id  INTEGER      NOT NULL REFERENCES Station(station_id),
    distance_km     NUMERIC(8,2),
    duration_min    INTEGER,
    carrier_name    VARCHAR(200),
    is_active       BOOLEAN      DEFAULT TRUE,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- Связь M:N маршрут–остановка
CREATE TABLE IF NOT EXISTS RouteStation (
    route_id    INTEGER NOT NULL REFERENCES Route(route_id)   ON DELETE CASCADE,
    station_id  INTEGER NOT NULL REFERENCES Station(station_id) ON DELETE CASCADE,
    stop_order  INTEGER NOT NULL,
    arrival_offset_min  INTEGER DEFAULT 0,
    is_waypoint BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (route_id, station_id)
);

CREATE TABLE IF NOT EXISTS Trip (
    trip_id     SERIAL      PRIMARY KEY,
    route_id    INTEGER     NOT NULL REFERENCES Route(route_id),
    departure_dt TIMESTAMP  NOT NULL,
    arrival_dt  TIMESTAMP,
    vehicle_type VARCHAR(50),
    capacity    INTEGER,
    status      VARCHAR(20) DEFAULT 'scheduled'
                CHECK (status IN ('scheduled','active','completed','cancelled')),
    created_at  TIMESTAMP   DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Факты (транзакционные данные)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS PassengerCount (
    count_id    SERIAL      PRIMARY KEY,
    trip_id     INTEGER     NOT NULL REFERENCES Trip(trip_id),
    period_date DATE        NOT NULL,
    passengers  INTEGER     NOT NULL CHECK (passengers >= 0),
    source      VARCHAR(50) DEFAULT 'manual'
                CHECK (source IN ('manual','ticket_system','sensor','import','synthetic')),
    created_at  TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_passengercount_trip_period ON PassengerCount(trip_id, period_date);
CREATE INDEX IF NOT EXISTS idx_passengercount_period ON PassengerCount(period_date);

-- -----------------------------------------------------------------------------
-- Внешние факторы
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ExternalFactor (
    factor_id   SERIAL       PRIMARY KEY,
    name        VARCHAR(150) NOT NULL,
    factor_type VARCHAR(50)  NOT NULL
                CHECK (factor_type IN ('economic','demographic','seasonal','infrastructure','other')),
    description TEXT,
    unit        VARCHAR(50),
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- Связь M:N поездка–фактор (значение фактора для конкретной поездки)
CREATE TABLE IF NOT EXISTS TripFactor (
    trip_id     INTEGER  NOT NULL REFERENCES Trip(trip_id)         ON DELETE CASCADE,
    factor_id   INTEGER  NOT NULL REFERENCES ExternalFactor(factor_id) ON DELETE CASCADE,
    period_date DATE     NOT NULL,
    value       NUMERIC(15,4) NOT NULL,
    PRIMARY KEY (trip_id, factor_id, period_date)
);

-- -----------------------------------------------------------------------------
-- Модели прогнозирования
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ForecastModel (
    model_id    SERIAL       PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    model_type  VARCHAR(50)  NOT NULL
                CHECK (model_type IN ('SARIMA','Prophet','LSTM','XGBoost','Ensemble')),
    version     VARCHAR(20)  DEFAULT '1.0',
    hyperparams JSONB,
    created_at  TIMESTAMP    DEFAULT NOW(),
    updated_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS Forecast (
    forecast_id     SERIAL      PRIMARY KEY,
    model_id        INTEGER     NOT NULL REFERENCES ForecastModel(model_id),
    route_id        INTEGER     NOT NULL REFERENCES Route(route_id),
    forecast_date   DATE        NOT NULL,
    horizon_months  INTEGER     NOT NULL CHECK (horizon_months IN (1,3,6,12)),
    passengers_pred INTEGER     NOT NULL CHECK (passengers_pred >= 0),
    ci_lower        INTEGER,
    ci_upper        INTEGER,
    created_at      TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecast_route_date ON Forecast(route_id, forecast_date);

CREATE TABLE IF NOT EXISTS ModelMetrics (
    metric_id   SERIAL      PRIMARY KEY,
    model_id    INTEGER     NOT NULL REFERENCES ForecastModel(model_id),
    route_id    INTEGER     NOT NULL REFERENCES Route(route_id),
    eval_date   DATE        NOT NULL,
    horizon     INTEGER,
    mae         NUMERIC(10,4),
    rmse        NUMERIC(10,4),
    mape        NUMERIC(8,4),
    r2          NUMERIC(6,4),
    created_at  TIMESTAMP   DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Пользователи и роли (RBAC)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS AppUser (
    user_id     SERIAL      PRIMARY KEY,
    username    VARCHAR(100) NOT NULL UNIQUE,
    email       VARCHAR(200) NOT NULL UNIQUE,
    role        VARCHAR(20)  NOT NULL DEFAULT 'dispatcher'
                CHECK (role IN ('planner','dispatcher','analyst','admin')),
    region_id   INTEGER      REFERENCES Region(region_id),
    is_active   BOOLEAN      DEFAULT TRUE,
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS Report (
    report_id   SERIAL      PRIMARY KEY,
    user_id     INTEGER     NOT NULL REFERENCES AppUser(user_id),
    report_type VARCHAR(50) NOT NULL
                CHECK (report_type IN ('weekly','monthly','quarterly','annual','custom')),
    parameters  JSONB,
    file_path   TEXT,
    created_at  TIMESTAMP   DEFAULT NOW()
);

-- =============================================================================
-- Индексы для аналитических запросов
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_trip_route_departure ON Trip(route_id, departure_dt);
CREATE INDEX IF NOT EXISTS idx_route_active ON Route(is_active);
CREATE INDEX IF NOT EXISTS idx_forecast_model_horizon ON Forecast(model_id, horizon_months);
CREATE INDEX IF NOT EXISTS idx_metrics_model_route ON ModelMetrics(model_id, route_id);
