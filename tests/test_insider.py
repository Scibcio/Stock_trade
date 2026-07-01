"""
Gate tests for insider.py (edge #1 - SEC EDGAR Form 4 parsing).
Offline: exercises the XML parser on a sample filing (no network).
"""

import insider

SAMPLE = """<ownershipDocument>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>60</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml():
    txns = insider.parse_form4_xml(SAMPLE)
    assert len(txns) == 2
    buy, sell = txns
    assert buy["code"] == "P" and buy["shares"] == 1000.0 and buy["price"] == 50.5 and buy["ad"] == "A"
    assert sell["code"] == "S" and sell["shares"] == 200.0 and sell["ad"] == "D"


def test_to_float_guards_whitespace():
    # the exact bug we hit: a whitespace value must become 0.0, not crash
    assert insider._to_float("\n   ") == 0.0
    assert insider._to_float("12.5") == 12.5
    assert insider._to_float(None) == 0.0
