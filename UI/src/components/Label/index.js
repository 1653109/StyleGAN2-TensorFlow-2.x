import { connect } from 'react-redux'

import Label from "./Label"
import { changeLabel } from '../ZGenerator/actions'

export default connect(
  ({ zGenerator: { label } }) => ({ label }),
  ({ changeLabel })
)(Label)
