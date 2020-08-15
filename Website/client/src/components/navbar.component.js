import React,{Component} from 'react';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';

export default class NavbarComponent extends Component{
    render(){
        return (
            <Navbar collapseOnSelect expand="lg" bg="dark" variant="dark">
                <Navbar.Brand href="#home">VSM</Navbar.Brand>
                <Navbar.Toggle aria-controls="responsive-navbar-nav" />
                <Navbar.Collapse id="responsive-navbar-nav">
                    <Nav className="mr-auto">
                        <Nav.Link href="#howItWorks">How It Works</Nav.Link>
                        <Nav.Link href="https://www.investopedia.com/articles/basics/06/invest1000.asp">Blog</Nav.Link>
                    </Nav>
                    <Nav>
                        <Nav.Link href="/signUp">Sign Up</Nav.Link>
                        <Nav.Link eventKey={2} href="/signIn">
                            Sign In
                        </Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        );
    }
}