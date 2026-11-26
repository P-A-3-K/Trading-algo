//+------------------------------------------------------------------+
//| QuantLabORB.mq5 - opening range breakout, the QuantLab live rules |
//|                                                                    |
//| One copy per chart:  US100.cash -> Session US,                     |
//|                      GER40.cash -> Session Europe.                 |
//| Same rules as quantlab/live.py: range from 1-minute candles, buy   |
//| stop above / sell stop below, one cancels the other, stop at the   |
//| other side (or middle), optional target, flat before the close,    |
//| size = equity x weight x multiplier x min(1, risk% / stop%),       |
//| daily + total loss guards shared by all copies, demo-only.         |
//| Works on MetaQuotes' VPS and in the Strategy Tester.               |
//+------------------------------------------------------------------+
#property copyright "QuantLab"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include "QuantLabCalendar.mqh"

enum QL_SESSION
  {
   SESSION_US = 0,      // US: 9:30-16:00 New York (US100, US500)
   SESSION_EUROPE = 1,  // Europe: 9:00-17:30 Frankfurt (GER40)
   SESSION_LONDON = 2,  // London: 8:00-16:30 (UK100)
   SESSION_TOKYO = 3    // Tokyo: 9:00-15:00 (JP225)
  };

input group "Strategy"
input QL_SESSION Session        = SESSION_US;  // Session
input int        RangeMinutes   = 30;          // Opening range (minutes)
input double     RR             = 0.0;         // Take profit, reward:risk (0 = none)
input bool       StopAtMiddle   = false;       // Stop at the middle of the range (false = other side)
input bool       AllowShort     = true;        // Also trade downside breaks
input double     RiskPct        = 1.0;         // Risk per trade (% of the leg's money)

input group "Size (your plan)"
input double     Weight             = 0.5;     // This leg's share of the account (0.5 = 50/50 portfolio)
input double     Multiplier         = 1.0;     // Plan size (2 = challenge at 2x)
input double     CushionMultiplier  = 0.0;     // Size once the account is down (0 = off)
input double     CushionDrawdownPct = 4.0;     // ... when equity is this % below the starting balance

input group "Safety"
input double     StartBalance       = 0.0;     // Starting balance (0 = balance on the first run)
input double     DailyLossGuardPct  = 3.5;     // Pause for the day after this loss (% of day-start equity)
input double     TotalLossGuardPct  = 8.0;     // Stop for good this % below the starting balance
input int        CloseMinutesBefore = 2;       // Flatten this many minutes before the session close
input int        LateMinutes        = 5;       // Skip the day if started this long after the range
input bool       AllowReal          = false;   // Allow a REAL-money account
input long       MagicNumber        = 0;       // Magic number (0 = automatic per session)
input int        ServerMinusNewYorkHours = 7;  // Broker clock minus New York clock (FTMO: 7)

CTrade trade;
long   g_magic;
string g_tag;
int    g_zone;            // 0 New York, 1 Frankfurt, 2 London, 3 Tokyo
int    g_open, g_close;   // session minutes on the local clock

// today's state (also kept in global variables so a restart doesn't trade twice)
int    g_day = 0;         // yyyymmdd on the session's clock
int    g_phase = 0;       // 0 waiting for the range, 1 orders out, 2 in a trade, 3 done for today
double g_hi = 0, g_lo = 0;
int    g_placed = 0;      // pending orders placed today
bool   g_longDone = false, g_shortDone = false;
int    g_fails = 0;
bool   g_warnedCalendar = false;

//+------------------------------------------------------------------+
//| Time zones (MQL5 has no time-zone database: DST rules by hand)    |
//+------------------------------------------------------------------+
datetime MakeTime(int y, int mo, int d, int h, int mi)
  {
   MqlDateTime s;
   ZeroMemory(s);
   s.year = y; s.mon = mo; s.day = d; s.hour = h; s.min = mi; s.sec = 0;
   return StructToTime(s);
  }

int DowOf(datetime t)  { MqlDateTime s; TimeToStruct(t, s); return s.day_of_week; }
int YearOf(datetime t) { MqlDateTime s; TimeToStruct(t, s); return s.year; }

datetime NthSunday(int y, int mo, int n)
  {
   datetime first = MakeTime(y, mo, 1, 0, 0);
   int d = 1 + (7 - DowOf(first)) % 7 + 7 * (n - 1);
   return MakeTime(y, mo, d, 0, 0);
  }

datetime LastSunday(int y, int mo)
  {
   int ny = y, nm = mo + 1;
   if(nm > 12) { nm = 1; ny++; }
   datetime last = MakeTime(ny, nm, 1, 0, 0) - 86400;
   return last - DowOf(last) * 86400;
  }

// US: 2nd Sunday of March 2:00 EST (7:00 UTC) to 1st Sunday of November 2:00 EDT (6:00 UTC)
bool UsDst(datetime utc)
  {
   int y = YearOf(utc);
   return utc >= NthSunday(y, 3, 2) + 7 * 3600 && utc < NthSunday(y, 11, 1) + 6 * 3600;
  }

// EU: last Sunday of March 1:00 UTC to last Sunday of October 1:00 UTC
bool EuDst(datetime utc)
  {
   int y = YearOf(utc);
   return utc >= LastSunday(y, 3) + 3600 && utc < LastSunday(y, 10) + 3600;
  }

int ZoneOffset(int zone, datetime utc)     // local clock minus UTC, seconds
  {
   switch(zone)
     {
      case 0: return UsDst(utc) ? -4 * 3600 : -5 * 3600;
      case 1: return EuDst(utc) ? 2 * 3600 : 3600;
      case 2: return EuDst(utc) ? 3600 : 0;
      case 3: return 9 * 3600;
     }
   return 0;
  }

datetime UtcToZone(int zone, datetime utc)   { return utc + ZoneOffset(zone, utc); }
datetime ZoneToUtc(int zone, datetime local) { datetime u = local - ZoneOffset(zone, local); return local - ZoneOffset(zone, u); }
datetime ServerToUtc(datetime s)              { return ZoneToUtc(0, s - ServerMinusNewYorkHours * 3600); }
datetime UtcToServer(datetime u)              { return UtcToZone(0, u) + ServerMinusNewYorkHours * 3600; }
datetime LocalToServer(datetime local)        { return UtcToServer(ZoneToUtc(g_zone, local)); }

int Ymd(datetime t) { MqlDateTime s; TimeToStruct(t, s); return s.year * 10000 + s.mon * 100 + s.day; }

//+------------------------------------------------------------------+
//| Holidays (from QuantLabCalendar.mqh)                              |
//+------------------------------------------------------------------+
bool InList(const int &a[], int v)
  {
   for(int i = 0; i < ArraySize(a); i++)
      if(a[i] == v) return true;
   return false;
  }

int EarlyFrom(const int &days[], const int &mins[], int v)
  {
   for(int i = 0; i < ArraySize(days); i++)
      if(days[i] == v) return mins[i];
   return -1;
  }

bool IsClosed(int ymd)
  {
   switch(g_zone)
     {
      case 0: return InList(QL_US_CLOSED, ymd);
      case 1: return InList(QL_EU_CLOSED, ymd);
      case 2: return InList(QL_UK_CLOSED, ymd);
      case 3: return InList(QL_JP_CLOSED, ymd);
     }
   return false;
  }

int CloseMinute(int ymd)   // the session's close today (early closes included)
  {
   int m = -1;
   switch(g_zone)
     {
      case 0: m = EarlyFrom(QL_US_EARLY, QL_US_EARLY_MIN, ymd); break;
      case 1: m = EarlyFrom(QL_EU_EARLY, QL_EU_EARLY_MIN, ymd); break;
      case 2: m = EarlyFrom(QL_UK_EARLY, QL_UK_EARLY_MIN, ymd); break;
      case 3: m = EarlyFrom(QL_JP_EARLY, QL_JP_EARLY_MIN, ymd); break;
     }
   return m > 0 ? m : g_close;
  }

//+------------------------------------------------------------------+
//| Journal and saved state                                           |
//+------------------------------------------------------------------+
void Log(string what) { Print(g_tag, what); }

string GV(string key)  { return "QL_" + IntegerToString(g_magic) + "_" + key; }

void SaveState()
  {
   GlobalVariableSet(GV("day"), g_day);
   GlobalVariableSet(GV("ph"), g_phase);
   GlobalVariableSet(GV("hi"), g_hi);
   GlobalVariableSet(GV("lo"), g_lo);
   GlobalVariableSet(GV("np"), g_placed);
   GlobalVariableSet(GV("ld"), g_longDone ? 1 : 0);
   GlobalVariableSet(GV("sd"), g_shortDone ? 1 : 0);
  }

bool LoadState(int day)
  {
   if(!GlobalVariableCheck(GV("day")) || (int)GlobalVariableGet(GV("day")) != day) return false;
   g_phase = (int)GlobalVariableGet(GV("ph"));
   g_hi = GlobalVariableGet(GV("hi"));
   g_lo = GlobalVariableGet(GV("lo"));
   g_placed = (int)GlobalVariableGet(GV("np"));
   g_longDone = GlobalVariableGet(GV("ld")) > 0;
   g_shortDone = GlobalVariableGet(GV("sd")) > 0;
   return true;
  }

//+------------------------------------------------------------------+
//| Orders and positions of this copy                                 |
//+------------------------------------------------------------------+
bool HasPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == g_magic)
         return true;
     }
   return false;
  }

int CountPending()
  {
   int n = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong t = OrderGetTicket(i);
      if(t > 0 && OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == g_magic)
         n++;
     }
   return n;
  }

void CancelPending()
  {
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      ulong t = OrderGetTicket(i);
      if(t > 0 && OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == g_magic)
         if(!trade.OrderDelete(t))
            Log("could not cancel order " + IntegerToString((long)t) + ": " + trade.ResultRetcodeDescription());
     }
  }

void ClosePositions(string why)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == g_magic)
        {
         if(trade.PositionClose(t))
            Log("closed (" + why + ") at " + DoubleToString(trade.ResultPrice(), _Digits));
         else
            Log("could not close position " + IntegerToString((long)t) + ": " + trade.ResultRetcodeDescription());
        }
     }
  }

void Flatten(string why) { CancelPending(); ClosePositions(why); }

//+------------------------------------------------------------------+
//| Sizing                                                            |
//+------------------------------------------------------------------+
double StartBal()
  {
   if(StartBalance > 0) return StartBalance;
   if(!GlobalVariableCheck("QL_START")) GlobalVariableSet("QL_START", AccountInfoDouble(ACCOUNT_BALANCE));
   return GlobalVariableGet("QL_START");
  }

double SizeMultiplier(double equity)
  {
   if(CushionMultiplier > 0 && equity <= StartBal() * (1 - CushionDrawdownPct / 100.0))
      return CushionMultiplier;
   return Multiplier;
  }

double LotsFor(double entry, double stop)
  {
   double risk = MathAbs(entry - stop);
   if(risk <= 0 || entry <= 0) return 0;
   double frac = MathMin(1.0, RiskPct / 100.0 / (risk / entry));
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double notional = equity * Weight * SizeMultiplier(equity) * frac;
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tv <= 0 || ts <= 0) return 0;
   double perLot = entry * tv / ts;                    // one lot's value in account currency
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) step = 0.01;
   double lots = MathFloor(notional / perLot / step + 1e-9) * step;
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(vmax > 0) lots = MathMin(lots, vmax);
   if(lots < MathMax(vmin, step)) return 0;
   int digits = (int)MathMax(0, MathCeil(-MathLog10(step) - 1e-9));
   return NormalizeDouble(lots, digits);
  }

double RoundTick(double p)
  {
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(ts <= 0) return NormalizeDouble(p, _Digits);
   return NormalizeDouble(MathRound(p / ts) * ts, _Digits);
  }

//+------------------------------------------------------------------+
//| Guards (shared by every copy on the account)                      |
//+------------------------------------------------------------------+
bool GuardsOk(datetime utc)
  {
   if(GlobalVariableCheck("QL_HALT") && GlobalVariableGet("QL_HALT") > 0)
     {
      if(HasPosition() || CountPending() > 0) Flatten("halted");
      return false;
     }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   string pday = IntegerToString(Ymd(UtcToZone(1, utc)));       // FTMO's day: Prague (CE(S)T)
   string dse = "QL_DSE_" + pday;
   if(!GlobalVariableCheck(dse)) GlobalVariableSet(dse, equity);
   double start = StartBal();
   if(equity <= start * (1 - TotalLossGuardPct / 100.0))
     {
      GlobalVariableSet("QL_HALT", 1);
      Log(StringFormat("TOTAL LOSS GUARD: equity %.2f is %.1f%% below the starting balance %.2f. Stopped for good "
                       "(delete the global variable QL_HALT to restart).", equity, TotalLossGuardPct, start));
      Flatten("total loss guard");
      return false;
     }
   string pause = "QL_PAUSE_" + pday;
   if(GlobalVariableCheck(pause))
     {
      if(HasPosition() || CountPending() > 0) Flatten("paused today");
      return false;
     }
   if(equity <= GlobalVariableGet(dse) * (1 - DailyLossGuardPct / 100.0))
     {
      GlobalVariableSet(pause, 1);
      Log(StringFormat("DAILY LOSS GUARD: equity %.2f is down %.1f%% today. Paused until tomorrow.", equity, DailyLossGuardPct));
      Flatten("daily loss guard");
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Today's routine                                                   |
//+------------------------------------------------------------------+
void NewDay(int day)
  {
   if(HasPosition() || CountPending() > 0)
     {
      Log("leftover orders/positions from an earlier day: cleaning up");
      Flatten("leftover");
     }
   g_day = day;
   if(!LoadState(day))
     {
      g_phase = 0; g_hi = 0; g_lo = 0; g_placed = 0;
      g_longDone = false; g_shortDone = !AllowShort;
      SaveState();
     }
   g_fails = 0;
  }

bool MeasureRange(datetime dayStart, int rangeEnd)
  {
   datetime from = LocalToServer(dayStart + g_open * 60);
   datetime to = LocalToServer(dayStart + rangeEnd * 60) - 60;      // last 1-minute candle inside the range
   MqlRates r[];
   int n = CopyRates(_Symbol, PERIOD_M1, from, to, r);
   if(n < 0) return false;                                           // history still loading: try again
   if(n < RangeMinutes * 0.8)
     {
      Log(StringFormat("range incomplete (%d of %d one-minute candles): no trade today", n, RangeMinutes));
      g_phase = 3;
      return true;
     }
   double hi = r[0].high, lo = r[0].low;
   for(int i = 1; i < n; i++) { hi = MathMax(hi, r[i].high); lo = MathMin(lo, r[i].low); }
   if(hi <= lo) { g_phase = 3; return true; }
   g_hi = hi; g_lo = lo; g_phase = 1;
   Log(StringFormat("range %s-%s  high %s  low %s", TimeToString(from, TIME_MINUTES), TimeToString(to + 60, TIME_MINUTES),
                    DoubleToString(hi, _Digits), DoubleToString(lo, _Digits)));
   return true;
  }

// Place (or retry) each side: a stop order beyond the range, or at market if already broken out
void Arm()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK), bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0 || bid <= 0) return;
   double mid = (g_hi + g_lo) / 2;
   double tsize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(!g_longDone)
     {
      double sl = RoundTick(StopAtMiddle ? mid : g_lo);
      bool now = ask >= g_hi;
      double entry = now ? ask : RoundTick(g_hi + (tsize > 0 ? tsize : _Point));   // strictly above the high
      double tp = RR > 0 ? RoundTick(entry + RR * (entry - sl)) : 0;
      double lots = LotsFor(entry, sl);
      if(now && bid <= sl) { g_longDone = true; Log("long: price already past the stop, skipped"); }
      else if(lots <= 0) { g_longDone = true; Log("long: size below the minimum lot, skipped"); }
      else if(now ? trade.Buy(lots, _Symbol, 0, sl, tp, "QL ORB long")
                  : trade.BuyStop(lots, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, "QL ORB long"))
        {
         g_longDone = true;
         if(!now) g_placed++;
         Log(StringFormat("long %s %.2f lots at %s  stop %s  target %s", now ? "market" : "buy stop", lots,
                          DoubleToString(now ? trade.ResultPrice() : entry, _Digits), DoubleToString(sl, _Digits),
                          tp > 0 ? DoubleToString(tp, _Digits) : "none"));
        }
      else if(g_fails++ < 3) Log("long order refused: " + trade.ResultRetcodeDescription() + " (retrying)");
     }
   if(!g_shortDone && !HasPosition())
     {
      double sl = RoundTick(StopAtMiddle ? mid : g_hi);
      bool now = bid <= g_lo;
      double entry = now ? bid : RoundTick(g_lo - (tsize > 0 ? tsize : _Point));
      double tp = RR > 0 ? RoundTick(entry - RR * (sl - entry)) : 0;
      double lots = LotsFor(entry, sl);
      if(now && ask >= sl) { g_shortDone = true; Log("short: price already past the stop, skipped"); }
      else if(lots <= 0) { g_shortDone = true; Log("short: size below the minimum lot, skipped"); }
      else if(now ? trade.Sell(lots, _Symbol, 0, sl, tp, "QL ORB short")
                  : trade.SellStop(lots, entry, _Symbol, sl, tp, ORDER_TIME_GTC, 0, "QL ORB short"))
        {
         g_shortDone = true;
         if(!now) g_placed++;
         Log(StringFormat("short %s %.2f lots at %s  stop %s  target %s", now ? "market" : "sell stop", lots,
                          DoubleToString(now ? trade.ResultPrice() : entry, _Digits), DoubleToString(sl, _Digits),
                          tp > 0 ? DoubleToString(tp, _Digits) : "none"));
        }
      else if(g_fails++ < 3) Log("short order refused: " + trade.ResultRetcodeDescription() + " (retrying)");
     }
  }

void Status(string what)
  {
   if(MQLInfoInteger(MQL_TESTER) && !MQLInfoInteger(MQL_VISUAL_MODE)) return;
   Comment("QuantLab ORB ", EnumToString(Session), "  magic ", g_magic, "\n", IntegerToString(g_day), ": ", what,
           g_hi > 0 ? StringFormat("\nrange %s - %s", DoubleToString(g_lo, _Digits), DoubleToString(g_hi, _Digits)) : "");
  }

void Poll()
  {
   datetime server = TimeTradeServer();
   datetime utc = ServerToUtc(server);
   datetime local = UtcToZone(g_zone, utc);
   long secs = (long)local;
   datetime dayStart = (datetime)(secs - secs % 86400);
   int day = Ymd(local);
   int minute = (int)((local - dayStart) / 60);

   if(day != g_day) NewDay(day);
   if(!GuardsOk(utc)) { Status("paused / halted by a loss guard"); return; }

   int dow = DowOf(local);
   if(dow == 0 || dow == 6) { Status("weekend"); return; }
   if(day / 10000 > 2030 && !g_warnedCalendar)
     {
      g_warnedCalendar = true;
      Log("the holiday calendar ends in 2030: run  python -m quantlab.mql5 calendar  and reinstall");
     }
   if(IsClosed(day))
     {
      if(g_phase != 3) { Log("exchange holiday today: no trade"); g_phase = 3; SaveState(); }
      Status("holiday");
      return;
     }
   int rangeEnd = g_open + RangeMinutes;
   int flatAt = CloseMinute(day) - CloseMinutesBefore;
   int before = g_phase;

   if(g_phase == 0 && minute >= rangeEnd)
     {
      if(minute >= flatAt) g_phase = 3;
      else if(minute > rangeEnd + LateMinutes) { Log("too late today (started after the range + " + IntegerToString(LateMinutes) + " min)"); g_phase = 3; }
      else MeasureRange(dayStart, rangeEnd);
     }
   if(g_phase == 1 || g_phase == 2)
     {
      if(minute >= flatAt)
        {
         Flatten("end of day");
         g_phase = 3;
        }
      else if(HasPosition())
        {
         if(CountPending() > 0) CancelPending();                    // one cancels the other
         if(g_phase != 2) Log("in the trade");
         g_phase = 2;
        }
      else if(g_phase == 2)
        {
         CancelPending();
         Log("trade closed (stop / target)");
         g_phase = 3;
        }
      else
        {
         int pending = CountPending();
         if(g_placed > 0 && pending < g_placed)                     // filled and closed between checks
           {
            CancelPending();
            Log("trade filled and closed between checks");
            g_phase = 3;
           }
         else
           {
            Arm();
            if(HasPosition()) g_phase = 2;
            else if(g_longDone && g_shortDone && CountPending() == 0) g_phase = 3;
           }
        }
     }
   if(g_phase != before) SaveState();
   string names[] = {"waiting for the range", "orders out", "in a trade", "done for today"};
   Status(names[g_phase]);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   g_zone = (int)Session;
   int opens[] = {570, 540, 480, 540};
   int closes[] = {960, 1050, 990, 900};
   g_open = opens[g_zone]; g_close = closes[g_zone];
   long h = 0;                                       // automatic: unique per symbol and session
   for(int i = 0; i < StringLen(_Symbol); i++) h = (h * 31 + StringGetCharacter(_Symbol, i)) % 1000;
   g_magic = MagicNumber > 0 ? MagicNumber : 7700000 + g_zone * 1000 + h;
   g_tag = "[QL " + _Symbol + " " + EnumToString(Session) + "] ";
   if(RangeMinutes < 1 || g_open + RangeMinutes >= g_close) { Log("bad RangeMinutes"); return INIT_PARAMETERS_INCORRECT; }

   bool tester = (bool)MQLInfoInteger(MQL_TESTER);
   long mode = AccountInfoInteger(ACCOUNT_TRADE_MODE);
   if(!tester && mode == ACCOUNT_TRADE_MODE_REAL && !AllowReal)
     {
      Alert(g_tag, "this is a REAL-money account: refused (set AllowReal = true to allow).");
      return INIT_FAILED;
     }
   trade.SetExpertMagicNumber(g_magic);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetDeviationInPoints(50);
   trade.LogLevel(LOG_LEVEL_ERRORS);

   if(!tester)
     {
      // check our clock rules against the broker's real clock
      datetime utc = TimeGMT();
      long expected = (long)(UtcToServer(utc) - utc), actual = (long)(TimeTradeServer() - utc);
      if(MathAbs((double)(expected - actual)) > 1800)
         Log(StringFormat("WARNING: the broker clock is UTC%+.1f h but the EA expects UTC%+.1f h. "
                          "Check ServerMinusNewYorkHours.", actual / 3600.0, expected / 3600.0));
      if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) Log("Algo Trading is OFF in the terminal: press the Algo Trading button");
     }
   Log(StringFormat("started: %s account, equity %.2f, start balance %.2f, weight %.2f, size x%.2f",
                    mode == ACCOUNT_TRADE_MODE_DEMO ? "demo" : mode == ACCOUNT_TRADE_MODE_CONTEST ? "contest" : "REAL",
                    AccountInfoDouble(ACCOUNT_EQUITY), StartBal(), Weight, Multiplier));
   EventSetTimer(5);
   Poll();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment("");
  }

void OnTick()  { Poll(); }
void OnTimer() { Poll(); }
//+------------------------------------------------------------------+
