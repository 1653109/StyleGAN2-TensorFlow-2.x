import React from 'react'
import { Breadcrumbs, Link } from '@material-ui/core'

const handleClick = (e) => {
  e.preventDefault()
  console.log('clicked header breadcrumps')
}

const Header = () => {
  return (
    <div style={{
      height: '5vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      border: '1px solid #e4e5e9',
      borderTop: 'none',
      background: '#FFFFFF'
    }}>
      <Breadcrumbs aria-label="breadcrumb">
        <Link color="inherit" href="/" onClick={handleClick}>
          Z Latents
      </Link>
        {/* <Link color="inherit" href="/w" onClick={handleClick}>
          W Latents
      </Link> */}
      </Breadcrumbs>
    </div>
  )
}

export default Header
