# Trend-Following-Strategy_2-

En trendfollowing strategi som liknar den första vi kollat på, men på högre timeframe och endast longs på index. Idén kommer inte från den föregående trend following strategin, utan den mean reversion strategin vi analyserat och byggt, som vi även kommer använda i våran 
portfölj.
Bullish Trend = EMA_fast över EMA_slow
Entry Signal = Föregående bar stänger under EMA_fast, nuvarande bar stänger ovanför.
Exit = Pris går under EMA_fast
