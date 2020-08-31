import React from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { Slider } from '@material-ui/core'
import { changeLatents, fetchImage } from '../ZGenerator/actions'

const valueText = (value) => {
  return `${Math.round(value * 100) / 100}`
}

const Latent = ({ idx }) => {
  const value = useSelector(state => state.zGenerator.latents[idx])
  const dispatch = useDispatch()

  const handleChange = (e, v) => {
    e.preventDefault()
    dispatch(changeLatents(idx, v))
  }

  const handleChangeCommited = (e, v) => {
    e.preventDefault()
    dispatch(fetchImage())
  }

  // console.log(`render idx: ${idx}!`)

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
      onChange={handleChange}
      onChangeCommitted={handleChangeCommited}
    />
  )
}

export default Latent
