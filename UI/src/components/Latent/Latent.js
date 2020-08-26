import React from 'react'
import { Slider } from '@material-ui/core'
// import { makeStyles } from '@material-ui/core/styles'
// import { useMemo } from 'react'
// import { render } from 'react-dom'

// const useStyles = makeStyles(() => ({
//   slider: {
//     width: 400
//   }
// }))

const valueText = (value) => {
  return `${Math.round(value * 100) / 100}`
}

/* const Latent = ({changeLatents, latents, idx, fetchImage}) => {
  const classes = useStyles()
  const value = useMemo(() => returnIdx(latents, idx), [latents, idx])

  const handleChangeCommited = (e, v, idx) => {
    changeLatents(idx, v)
    fetchImage()
  }

  console.log(`render idx: ${idx}!`)

  return (
    <Slider
      className={classes.slider}
      track={false}
      defaultValue={0}
      value={value}
      min={-5.7}
      max={5.7}
      step={0.01}
      aria-labelledby="track-false-slider"
      getAriaValueText={valueText}
      valueLabelDisplay="auto"
      onChange={(e, v) => changeLatents(idx, v)}
      onChangeCommitted={handleChangeCommited}
    />
  )
} */

class Latent extends React.Component {
  shouldComponentUpdate(nextProps, nextState) {
    const { latents, idx } = this.props
    if (latents[idx] === nextProps.latents[idx]) {
      return false
    }
    return true
  }

  render() {
    const { changeLatents, latents, idx, fetchImage } = this.props
    const value = latents[idx]
    console.log(`Latents ${idx} rendered!`)
    return (
      <Slider
        style={{ width: 400 }}
        track={false}
        defaultValue={0}
        value={value}
        min={-5.7}
        max={5.7}
        step={0.01}
        aria-labelledby="track-false-slider"
        getAriaValueText={valueText}
        valueLabelDisplay="auto"
        onChange={(e, v) => changeLatents(idx, v)}
        onChangeCommitted={(e, v) => {
          changeLatents(idx, v)
          fetchImage()
        }}
      />
    )
  }
}

export default Latent
