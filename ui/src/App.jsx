import React, { useState } from 'react';

export default function App() {
  const [seq, setSeq] = useState('');
  const [result, setResult] = useState(null);

  const explain = async () => {
    const res = await fetch('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequences: [seq] })
    });
    setResult(await res.json());
  };

  return (
    <div>
      <h1>MAP-ExPLoc</h1>
      <input value={seq} onChange={e => setSeq(e.target.value)} />
      <button onClick={explain}>Explain</button>
      <pre>{result && JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}
