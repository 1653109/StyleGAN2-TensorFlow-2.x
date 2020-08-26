import React from 'react'
import {
  FormGroup,
  FormControlLabel,
  // Checkbox,
  Button,
  Slider,
  Radio,
  RadioGroup,
  FormLabel,
  FormControl
} from '@material-ui/core'

const Label = ({ label, changeLabel }) => {

  const handleChange = (event) => {
    changeLabel(parseInt(event.target.value));
  };

  return (
    <FormControl component="fieldset">
      <FormLabel component="legend">Label</FormLabel>
      <RadioGroup aria-label="gender" row name="label" value={label} onChange={handleChange} >
        <FormControlLabel value={0} control={<Radio />} label="Legs" />
        <FormControlLabel value={1} control={<Radio />} label="Others" />
        <FormControlLabel value={2} control={<Radio />} label="Wings" />
        <FormControlLabel value={3} control={<Radio />} label="Faces" />
      </RadioGroup>
    </FormControl>
  )
}

export default Label
