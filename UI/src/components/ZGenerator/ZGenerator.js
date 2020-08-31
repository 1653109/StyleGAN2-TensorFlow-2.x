import React, { useEffect } from 'react'
import { Grid } from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import { Paper } from '@material-ui/core'
import Logo from '../Logo'
import Modify from '../Modify'
import Loading from '../Loading'
import Latent from '../Latent'
import Label from '../Label'

// const latents = Array(512).fill(0)

const useStyles = makeStyles(theme => ({
  root: {
    padding: '10px',
    // display: 'flex',
    // justifyContent: 'space-between'
  },
  slider: {
    width: 400
  },
  margin: {
    height: theme.spacing(3),
    width: theme.spacing(3)
  },
  latents: {
    padding: '35px 10px',
    display: 'inline-block',
    overflowY: 'scroll',
    height: '50vh'
  }
}))

const ZGenerator = ({ isFetchingInfo, fetchInfo, latentsDimensions }) => {
  useEffect(() => {
    fetchInfo()
  }, [])

  const classes = useStyles()

  let render

  let latents = Array(latentsDimensions).fill(0)

  if (isFetchingInfo) {
    render = <Loading />
  } else {
    render = (
      <Grid container spacing={3}>
        <Grid item xs={4}>
          <Logo modelName="Logo" descrition="Generate from Z latents" />
        </Grid>
        <Grid item xs={8}>
          <Modify />
          <div className={classes.margin} />
          <Paper elevation={2} className={classes.root}>
            <Label />
          </Paper>
          <div className={classes.margin} />
          <Paper elevation={2} className={classes.latents}>
            {
              latents.map((value, idx) => {
                return (
                  <div key={idx} style={{
                    display: 'inline-block',
                    margin: '0 6%'
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}>
                      <span style={{ width: '50px' }}>{idx + 1} .</span>
                      <Latent idx={idx} />
                    </div>
                  </div>
                )
              })
            }
          </Paper>
        </Grid>
      </Grid>
    )
  }

  console.log('zGenerator render!')

  return (
    <div style={{
      height: 'calc(95vh - 61px)',
      padding: '30px',
      overflow: 'hidden',
      position: 'relative'
    }}>
      {render}
    </div>
  )
}

export default ZGenerator