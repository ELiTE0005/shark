python testing.py 0xYourWalletAddressHere --save-plot balance.png --save-csv txs.csv --tokens




Connects to the Etherscan API.

Pulls:

Normal transactions (incoming/outgoing ETH).

Internal transactions (smart contract calls moving ETH).

(Optional) ERC-20 token transfers and NFT transfers.

📊 Processing

Calculates:

Balance over time (ETH) by walking through every transaction.

Incoming vs outgoing transaction counts.

Gas fees paid (total cumulative fees).

Largest transactions.

📈 Visualization

Creates a Matplotlib plot with:

ETH balance line (your wallet balance over time).

Bars for each transaction (positive = incoming, negative = outgoing).

Cumulative gas fees subplot.

Optional rolling average line to smooth fluctuations.
