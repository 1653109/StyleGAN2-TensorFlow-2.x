import React from 'react';
import ZGenerator from '../ZGenerator';
import Header from '../Header';

function App() {
  return (
    <div style={{
      background: '#7f7fd5',
      background: '-webkit-linear-gradient(to top, #7f7fd5, #86a8e7, #91eae4)',
      background: 'linear-gradient(to top, #7f7fd5, #86a8e7, #91eae4)'
    }}>
      <Header />
      <ZGenerator />
    </div>
  );
}

export default App;
