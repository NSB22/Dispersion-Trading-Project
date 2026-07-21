"""
Point-in-time S&P 500 universe construction.

Usage:
    from dispersion.data.universe import get_universe
    df = get_universe(db, "2020-03-31")
"""
import pandas as pd
import wrds


def get_universe(db: wrds.Connection, date: str, n: int = 100) -> pd.DataFrame:
    """Top-N S&P 500 constituents by market cap on `date`, point-in-time.

    Only names with a score-1 OptionMetrics link are kept, so each permno maps
    to exactly one secid.

    Columns: permno, secid, market_cap (USD on `date`), weight (cap weight
    renormalised within the top-N), rnk (rank by cap, 1 = largest).
    """
    query = f"""
    WITH members AS (
        -- point-in-time S&P 500 membership
        SELECT permno
        FROM crsp.dsp500list
        WHERE start <= '{date}' AND ending >= '{date}'
    ),
    capi AS (
        -- market cap = abs(prc) * shrout * 1000
        -- CRSP prc can be negative (it's a bid/ask midpoint)
        SELECT d.permno,
               ABS(d.prc) * d.shrout * 1000 AS market_cap
        FROM crsp.dsf d
        JOIN members m ON d.permno = m.permno
        WHERE d.date = '{date}'
          AND d.prc IS NOT NULL
          AND d.shrout IS NOT NULL
    ),
    ranked AS (
        -- ROW_NUMBER not RANK: with RANK a cap tie at rank N would let in >N rows.
        -- permno breaks ties deterministically.
        SELECT permno, market_cap,
               ROW_NUMBER() OVER (ORDER BY market_cap DESC, permno) AS rnk
        FROM capi
    ),
    top_n AS (
        SELECT * FROM ranked WHERE rnk <= {n}
    ),
    link AS (
        -- score=1 only. A permno can have several overlapping score-1 links;
        -- keep the latest edate so secid is unique per permno.
        SELECT DISTINCT ON (permno) permno, secid
        FROM wrdsapps_link_crsp_optionm.opcrsphist
        WHERE score = 1 AND sdate <= '{date}' AND edate >= '{date}'
        ORDER BY permno, edate DESC
    )
    SELECT t.permno,
           l.secid,
           t.market_cap,
           t.market_cap / SUM(t.market_cap) OVER () AS weight,
           t.rnk
    FROM top_n t
    LEFT JOIN link l ON t.permno = l.permno
    ORDER BY t.rnk
    """
    return db.raw_sql(query)
