import React,{Component} from 'react';
import '../index.css';

export default class Button extends Component{
    render(){
        return(
            <button className = 'btn' onClick={this.props.handleClick} type = {this.props.type} style = {this.props.style}>{this.props.label}</button>
        )
    }
}