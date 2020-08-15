const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const UserSchema = new Schema({
    name:{
        type:String,
        required: true
    },
    email: {
        type: String,
        required: true
    },
    password: {
        type: String,
        required: true
    },
    value:{
        type: Number,
        default: 0,
    },
    totalInvestment:{
        type:Number,
        default:0,
    },
    stocks:{
        type:Array,
        default:[]
    },
    marketWatch:{
        type:Array,
        default:[]
    }
});

module.exports = User = mongoose.model("user",UserSchema);