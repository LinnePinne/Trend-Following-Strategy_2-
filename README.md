# Trend-Following-Strategy_2-

En trendfollowing strategi som liknar den första vi kollat på, men på högre timeframe och endast longs på index. Idén kommer inte från den föregående trend following strategin, utan den mean reversion strategin vi analyserat och byggt, som vi även kommer använda i våran 
portfölj.
bullish_trend = ema_medium > ema_slow
cross = prev_ema_fast < ema_medium and ema_fast > ema_medium
long_entry_signal = bullish_trend and cross
Exit = ema_fast < ema_medium
