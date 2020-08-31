import React from 'react'
import {
  Paper,
  FormGroup,
  FormControlLabel,
  Button,
  Slider,
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles'
import PropTypes from 'prop-types'

const useStyles = makeStyles(theme => ({
  root: {
    padding: '10px'
  },
  slider: {
    width: 400
  },
  margin: {
    height: theme.spacing(3),
    width: theme.spacing(3)
  },
  latents: {
    padding: '10px',
    display: 'inline-block',
    overflowY: 'scroll',
    height: '50vh'
  }
}))

const marksTruncate = [
  {
    value: -1,
    label: '-1',
  },
  {
    value: 0,
    label: '0',
  },
  {
    value: 1,
    label: '1',
  }
]

const valueText = (value) => {
  return `${Math.round(value * 100) / 100}`
}

const Modify = ({ psi, changePsi, randomLatents, fetchImage }) => {
  const classes = useStyles()

  const handlePsiChange = (e, v) => {
    e.preventDefault()
    changePsi(v)
  }

  const handlePsiChangeCommitted = (e, v) => {
    e.preventDefault()
    fetchImage()
  }

  const handleRndBtn = (e) => {
    e.preventDefault()
    randomLatents()
  }

  return (
    <div>
      <Paper elevation={2}>
        <FormGroup row className={classes.root}>
          <Button variant="contained" color="primary" onClick={handleRndBtn}>
            Random
          </Button>
          <div className={classes.margin} />
          <FormControlLabel label="Truncation Ψ" control={<></>} />
          <Slider
            className={classes.slider}
            track={false}
            defaultValue={0}
            min={-1.5}
            max={1.5}
            step={0.001}
            marks={marksTruncate}
            aria-labelledby="track-false-slider"
            getAriaValueText={valueText}
            valueLabelDisplay={valueText}
            value={psi}
            onChange={handlePsiChange}
            onChangeCommitted={handlePsiChangeCommitted}
          />
        </FormGroup>
      </Paper>
    </div>
  )
}

Modify.propTypes = {
  latents: PropTypes.array
}

export default Modify
