//+------------------------------------------------------------------+
//| GDB_XAUUSD.mq5 — Gold Daily Breakout (research port, DEMO FIRST) |
//|                                                                  |
//| Module A: intraday volatility breakout, long only.               |
//|   At each new D1 bar: if yesterday's close > SMA(200) on D1,     |
//|   place a buy-stop at today's open + K * yesterday's range.      |
//|   Any A position / pending order is closed before day end.       |
//| Module B: Donchian swing breakout, long only.                    |
//|   At each new D1 bar: if yesterday's close > highest high of the |
//|   N bars before it AND > SMA(200), buy at market.                |
//|   Initial SL = entry - StopATR*ATR(14); trailing SL =            |
//|   highest high since entry - TrailATR*ATR(14), updated daily.    |
//| Parameters match run_gold.py. Attach to an XAUUSD chart (any TF).|
//+------------------------------------------------------------------+
#property strict
#property version "1.00"
#include <Trade/Trade.mqh>

input group "Module A — day-trade volatility breakout"
input bool   UseModuleA     = true;
input double A_K            = 0.7;    // breakout = open + K * prev range
input double A_Leverage     = 1.0;    // notional / equity for module A
input int    A_CloseHour    = 23;     // server hour to flatten A (set before rollover)
input int    A_CloseMinute  = 45;

input group "Module B — Donchian swing"
input bool   UseModuleB     = true;
input int    B_Channel      = 20;
input double B_StopATR      = 2.0;
input double B_TrailATR     = 2.0;
input double B_RiskPercent  = 1.0;    // % of equity risked per trade

input group "Common"
input int    TrendSMA       = 200;
input int    ATRPeriod      = 14;
input long   MagicA         = 26091;
input long   MagicB         = 26092;

CTrade   trade;
int      hSMA = INVALID_HANDLE, hATR = INVALID_HANDLE;
datetime lastBar = 0;

int OnInit()
{
   hSMA = iMA(_Symbol, PERIOD_D1, TrendSMA, 0, MODE_SMA, PRICE_CLOSE);
   hATR = iATR(_Symbol, PERIOD_D1, ATRPeriod);
   if(hSMA == INVALID_HANDLE || hATR == INVALID_HANDLE) return INIT_FAILED;
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   IndicatorRelease(hSMA);
   IndicatorRelease(hATR);
}

double Buf(int handle, int shift)
{
   double v[];
   if(CopyBuffer(handle, 0, shift, 1, v) != 1) return EMPTY_VALUE;
   return v[0];
}

// Lots for a given notional (USD) or for a given risk amount over a price distance.
double NormalizeLots(double lots)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   if(lots < vmin) return 0.0;
   return MathMin(lots, vmax);
}

double LotsForNotional(double notional, double price)
{
   double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   return NormalizeLots(notional / (price * contract));
}

double LotsForRisk(double riskMoney, double dist)
{
   double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(dist <= 0 || tickVal <= 0 || tickSz <= 0) return 0.0;
   return NormalizeLots(riskMoney / (dist / tickSz * tickVal));
}

bool HasPosition(long magic, ulong &ticket)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(PositionSelectByTicket(t) && PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == magic) { ticket = t; return true; }
   }
   return false;
}

void DeletePending(long magic)
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong t = OrderGetTicket(i);
      if(OrderSelect(t) && OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == magic)
         trade.OrderDelete(t);
   }
}

void FlattenA()
{
   DeletePending(MagicA);
   ulong t;
   if(HasPosition(MagicA, t)) { trade.SetExpertMagicNumber(MagicA); trade.PositionClose(t); }
}

void OnNewDay()
{
   double sma = Buf(hSMA, 1), atr = Buf(hATR, 1);
   double c1 = iClose(_Symbol, PERIOD_D1, 1);
   double h1 = iHigh(_Symbol, PERIOD_D1, 1), l1 = iLow(_Symbol, PERIOD_D1, 1);
   double o0 = iOpen(_Symbol, PERIOD_D1, 0);
   if(sma == EMPTY_VALUE || atr == EMPTY_VALUE || c1 == 0) return;
   bool uptrend = c1 > sma;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   // ---- Module A: new buy-stop for today
   if(UseModuleA)
   {
      FlattenA();
      if(uptrend)
      {
         double level = NormalizeDouble(o0 + A_K * (h1 - l1), digits);
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double lots = LotsForNotional(equity * A_Leverage, level);
         trade.SetExpertMagicNumber(MagicA);
         if(lots > 0)
         {
            if(level <= ask) trade.Buy(lots, _Symbol, 0, 0, 0, "GDB-A");   // already through level
            else trade.BuyStop(lots, level, _Symbol, 0, 0, ORDER_TIME_DAY, 0, "GDB-A");
         }
      }
   }

   // ---- Module B: trail existing position, or enter on a fresh breakout
   if(UseModuleB)
   {
      ulong t;
      trade.SetExpertMagicNumber(MagicB);
      if(HasPosition(MagicB, t))
      {
         datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
         int bars = iBarShift(_Symbol, PERIOD_D1, opened);
         int hiIdx = iHighest(_Symbol, PERIOD_D1, MODE_HIGH, bars + 1, 1);
         double best = MathMax(iHigh(_Symbol, PERIOD_D1, hiIdx), PositionGetDouble(POSITION_PRICE_OPEN));
         double newSL = NormalizeDouble(best - B_TrailATR * atr, digits);
         double curSL = PositionGetDouble(POSITION_SL);
         if(newSL > curSL + _Point && newSL < SymbolInfoDouble(_Symbol, SYMBOL_BID))
            trade.PositionModify(t, newSL, 0);
      }
      else
      {
         int idx = iHighest(_Symbol, PERIOD_D1, MODE_HIGH, B_Channel, 2);
         double chanHigh = iHigh(_Symbol, PERIOD_D1, idx);
         if(uptrend && c1 > chanHigh)
         {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double dist = B_StopATR * atr;
            double lots = LotsForRisk(equity * B_RiskPercent / 100.0, dist);
            if(lots > 0) trade.Buy(lots, _Symbol, ask, NormalizeDouble(ask - dist, digits), 0, "GDB-B");
         }
      }
   }
}

void OnTick()
{
   datetime bar = iTime(_Symbol, PERIOD_D1, 0);
   if(bar != lastBar && bar != 0)
   {
      lastBar = bar;
      OnNewDay();
   }
   if(UseModuleA)
   {
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      if(now.hour > A_CloseHour || (now.hour == A_CloseHour && now.min >= A_CloseMinute))
         FlattenA();
   }
}
