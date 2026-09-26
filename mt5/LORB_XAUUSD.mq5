//+------------------------------------------------------------------+
//| LORB_XAUUSD.mq5 — London Opening-Range Breakout, intraday only   |
//| Research port of run_intraday.py. TEST ON DEMO FIRST.            |
//|                                                                  |
//| Daily trend = yesterday's D1 close vs SMA(20) of D1 closes.      |
//| Range = high/low of the first 60 minutes from 07:00 London.      |
//| Uptrend   -> BuyStop at range high, SL at range low.             |
//| Downtrend -> SellStop at range low, SL at range high.            |
//| No take-profit. Pending order and position are closed at 20:00  |
//| London. One trade per day. Never holds overnight.                |
//|                                                                  |
//| Times are London time converted with ServerMinusLondon (hours).  |
//| Typical brokers on GMT+2/+3 (NY-close) servers: 2. Check yours:  |
//| server clock minus London clock, and review it around DST dates. |
//+------------------------------------------------------------------+
#property strict
#property version "1.00"
#include <Trade/Trade.mqh>

input int    ServerMinusLondon = 2;     // hours: server time - London time
input int    RangeStartLondon  = 7;     // range start hour (London)
input int    RangeMinutes      = 60;
input int    ExitHourLondon    = 20;    // flat hour (London)
input int    TrendSMA          = 20;    // 0 = no trend filter (both orders placed, OCO)
input double RiskPercent       = 0.25;  // % equity risked per trade (backtest: 0.25-0.5)
input double MaxLeverage       = 10.0;  // notional cap vs equity
input long   Magic             = 26093;

CTrade trade;
int    hSMA = INVALID_HANDLE;
int    lastSetupDay = -1;

int OnInit()
{
   if(TrendSMA > 0)
   {
      hSMA = iMA(_Symbol, PERIOD_D1, TrendSMA, 0, MODE_SMA, PRICE_CLOSE);
      if(hSMA == INVALID_HANDLE) return INIT_FAILED;
   }
   trade.SetExpertMagicNumber(Magic);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { if(hSMA != INVALID_HANDLE) IndicatorRelease(hSMA); }

int ServerMinuteOfDay()
{
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);
   return t.hour * 60 + t.min;
}

int LondonToServerMinute(int londonHour, int addMinutes)
{
   int m = (londonHour + ServerMinusLondon) * 60 + addMinutes;
   return ((m % 1440) + 1440) % 1440;
}

int TrendDir()
{
   if(TrendSMA <= 0) return 0;
   double v[];
   if(CopyBuffer(hSMA, 0, 1, 1, v) != 1) return 99;
   double c1 = iClose(_Symbol, PERIOD_D1, 1);
   return c1 > v[0] ? 1 : (c1 < v[0] ? -1 : 99);
}

double NormalizeLots(double lots)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   return lots < vmin ? 0.0 : MathMin(lots, vmax);
}

double LotsFor(double dist, double price)
{
   double eq       = AccountInfoDouble(ACCOUNT_EQUITY);
   double tickVal  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSz   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   if(dist <= 0 || tickVal <= 0 || tickSz <= 0) return 0.0;
   double byRisk = eq * RiskPercent / 100.0 / (dist / tickSz * tickVal);
   double byLev  = eq * MaxLeverage / (price * contract);
   return NormalizeLots(MathMin(byRisk, byLev));
}

bool HasOwnPosition(ulong &ticket)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(PositionSelectByTicket(t) && PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == Magic) { ticket = t; return true; }
   }
   return false;
}

int DeleteOwnPending()
{
   int n = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong t = OrderGetTicket(i);
      if(OrderSelect(t) && OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == Magic)
      { trade.OrderDelete(t); n++; }
   }
   return n;
}

void FlattenAll()
{
   DeleteOwnPending();
   ulong t;
   if(HasOwnPosition(t)) trade.PositionClose(t);
}

void PlaceRangeOrders()
{
   // range = M1 bars from RangeStart to RangeStart + RangeMinutes (server time, today)
   MqlDateTime d; TimeToStruct(TimeCurrent(), d);
   d.hour = 0; d.min = 0; d.sec = 0;
   datetime day0 = StructToTime(d);
   datetime from = day0 + LondonToServerMinute(RangeStartLondon, 0) * 60;
   datetime to   = from + RangeMinutes * 60 - 1;
   MqlRates r[];
   int n = CopyRates(_Symbol, PERIOD_M1, from, to, r);
   if(n < RangeMinutes / 2) return;  // not enough data (holiday / gap)
   double hi = r[0].high, lo = r[0].low;
   for(int i = 1; i < n; i++) { hi = MathMax(hi, r[i].high); lo = MathMin(lo, r[i].low); }
   double w = hi - lo;
   if(w <= 0) return;

   int dir = TrendDir();
   if(dir == 99) return;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK), bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   hi = NormalizeDouble(hi, digits); lo = NormalizeDouble(lo, digits);

   if(dir >= 0 && ask < hi)
   {
      double lots = LotsFor(w, hi);
      if(lots > 0) trade.BuyStop(lots, hi, _Symbol, lo, 0, ORDER_TIME_GTC, 0, "LORB buy");
   }
   if(dir <= 0 && bid > lo)
   {
      double lots = LotsFor(w, lo);
      if(lots > 0) trade.SellStop(lots, lo, _Symbol, hi, 0, ORDER_TIME_GTC, 0, "LORB sell");
   }
}

void OnTick()
{
   int now = ServerMinuteOfDay();
   int setup = LondonToServerMinute(RangeStartLondon, RangeMinutes);
   int flat  = LondonToServerMinute(ExitHourLondon, 0);
   MqlDateTime t; TimeToStruct(TimeCurrent(), t);

   // 1) flat time: close everything
   if(now >= flat) { FlattenAll(); return; }

   // 2) once per day after the range has formed: place the stop order(s)
   if(now >= setup && lastSetupDay != t.day_of_year)
   {
      lastSetupDay = t.day_of_year;
      ulong p;
      if(!HasOwnPosition(p)) PlaceRangeOrders();
   }

   // 3) one trade per day: once a position exists, remove the opposite pending order
   ulong pos;
   if(HasOwnPosition(pos)) DeleteOwnPending();
}
